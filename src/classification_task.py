import os
import csv
import time
import copy
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from config import CLASSIFICATION, ZONE_KERNEL_SPECS, GLOBAL_SIGMA
from kernels import get_torch_device, apply_global_gaussian_torch, kernel_bank_torch, apply_zonewise_pdk_hard_torch
from zone_generators import classification_score_torch, quantize_score_to_zone_torch
from utils import ensure_dir

DATASET_NAME = CLASSIFICATION['dataset'].upper()
DATA_ROOT = CLASSIFICATION['data_root']
TINY_IMAGENET_ROOT = CLASSIFICATION['tiny_imagenet_root']
DEVICE = CLASSIFICATION['device']
BATCH_SIZE = CLASSIFICATION['batch_size']
EPOCHS = CLASSIFICATION['epochs']
LR = CLASSIFICATION['lr']
WEIGHT_DECAY = CLASSIFICATION['weight_decay']
NUM_WORKERS = CLASSIFICATION['num_workers']
SAVE_ROOT = CLASSIFICATION['save_root']
ZONE_HIGH = CLASSIFICATION['zone_high_percentile']
ZONE_MID = CLASSIFICATION['zone_mid_percentile']
SCORE_WEIGHTS = CLASSIFICATION['score_weights']
ensure_dir(SAVE_ROOT)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        val_annot_path = os.path.join(root, 'val', 'val_annotations.txt')
        val_img_dir = os.path.join(root, 'val', 'images')
        with open(val_annot_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('	')
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                self.samples.append((os.path.join(val_img_dir, img_name), class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(dataset_name: str):
    if dataset_name == 'CIFAR10':
        image_size = 32
    elif dataset_name == 'STL10':
        image_size = 96
    elif dataset_name == 'TINYIMAGENET':
        image_size = 64
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    train_tfm = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tfm = transforms.Compose([transforms.ToTensor()])
    return train_tfm, test_tfm


def build_loaders():
    train_tfm, test_tfm = build_transforms(DATASET_NAME)
    if DATASET_NAME == 'CIFAR10':
        train_ds = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_tfm)
        test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tfm)
        num_classes = 10
    elif DATASET_NAME == 'STL10':
        train_ds = datasets.STL10(root=DATA_ROOT, split='train', download=True, transform=train_tfm)
        test_ds = datasets.STL10(root=DATA_ROOT, split='test', download=True, transform=test_tfm)
        num_classes = 10
    elif DATASET_NAME == 'TINYIMAGENET':
        train_root = os.path.join(TINY_IMAGENET_ROOT, 'train')
        train_ds = datasets.ImageFolder(root=train_root, transform=train_tfm)
        test_ds = TinyImageNetValDataset(root=TINY_IMAGENET_ROOT, class_to_idx=train_ds.class_to_idx, transform=test_tfm)
        num_classes = len(train_ds.class_to_idx)
    else:
        raise ValueError(f'Unsupported dataset: {DATASET_NAME}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader, num_classes


def build_classifier(num_classes: int, dataset_name: str):
    model = models.shufflenet_v2_x0_5(weights=None)
    model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer_and_scheduler(model_params):
    optimizer = torch.optim.AdamW(model_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    return optimizer, scheduler


def preprocess_batch(x: torch.Tensor, mode: str, kernel_bank=None):
    if mode == 'raw':
        return x

    global_x = apply_global_gaussian_torch(x, GLOBAL_SIGMA['classification'])
    if mode == 'global':
        return global_x

    if mode == 'pdk':
        score = classification_score_torch(x, global_x, SCORE_WEIGHTS['residual'], SCORE_WEIGHTS['gradient'])
        zone = quantize_score_to_zone_torch(score, ZONE_HIGH, ZONE_MID)
        return apply_zonewise_pdk_hard_torch(x, zone, kernel_bank)

    raise ValueError(f'Unknown mode: {mode}')


@torch.no_grad()
def evaluate_classifier(clf, loader, device, mode: str, kernel_bank=None):
    clf.eval()
    total = 0
    correct = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_in = preprocess_batch(x, mode, kernel_bank)
        logits = clf(x_in)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    elapsed = time.time() - t0
    return correct / total, elapsed


def print_metric_guide():
    print("\n=== Metric Guide ===")
    print("Train CE   : mean cross-entropy loss over all training mini-batches in the current epoch.")
    print("Train Acc  : training accuracy of the current epoch.")
    print("Test Acc   : test-set accuracy after the current epoch.")
    print("Best Test  : best test accuracy observed so far.")
    print("LR         : learning rate after the scheduler step.")
    print("====================\n")


def train_single_mode(clf, train_loader, test_loader, device, mode: str, kernel_bank=None):
    optimizer, scheduler = build_optimizer_and_scheduler(clf.parameters())
    best_acc = 0.0
    best_state = None
    logs = []

    for epoch in range(1, EPOCHS + 1):
        clf.train()
        ce_vals = []
        train_total = 0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_in = preprocess_batch(x, mode, kernel_bank)

            optimizer.zero_grad()
            logits = clf(x_in)
            ce = F.cross_entropy(logits, y)
            ce.backward()
            optimizer.step()

            ce_vals.append(float(ce.detach().cpu()))
            pred = logits.argmax(dim=1)
            train_total += y.numel()
            train_correct += (pred == y).sum().item()

        scheduler.step()
        train_ce = float(sum(ce_vals) / len(ce_vals))
        train_acc = train_correct / train_total
        test_acc, _ = evaluate_classifier(clf, test_loader, device, mode, kernel_bank)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(clf.state_dict())

        current_lr = optimizer.param_groups[0]['lr']
        logs.append({
            'epoch': epoch,
            'train_ce': train_ce,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_test_acc': best_acc,
            'lr': current_lr,
        })

        print(
            f"[{mode.upper()}] Epoch {epoch:03d} | "
            f"Train CE={train_ce:.5f} | "
            f"Train Acc={train_acc:.4f} | "
            f"Test Acc={test_acc:.4f} | "
            f"Best Test={best_acc:.4f} | "
            f"LR={current_lr:.6e}"
        )

    if best_state is not None:
        clf.load_state_dict(best_state)
    return best_acc, logs


def run():
    device = get_torch_device(DEVICE)

    print(f"\n=== Classification Task ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("Classifier: ShuffleNetV2 x0.5")
    print("Scheduler: CosineAnnealingLR")
    print_metric_guide()

    train_loader, test_loader, num_classes = build_loaders()

    print(f"Num classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    kernel_bank = kernel_bank_torch(ZONE_KERNEL_SPECS, in_channels=3, device=device)

    raw_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    raw_best_acc, raw_logs = train_single_mode(raw_clf, train_loader, test_loader, device, mode='raw', kernel_bank=None)
    raw_final_acc, raw_eval_time = evaluate_classifier(raw_clf, test_loader, device, mode='raw', kernel_bank=None)

    global_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    global_best_acc, global_logs = train_single_mode(global_clf, train_loader, test_loader, device, mode='global', kernel_bank=None)
    global_final_acc, global_eval_time = evaluate_classifier(global_clf, test_loader, device, mode='global', kernel_bank=None)

    pdk_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    pdk_best_acc, pdk_logs = train_single_mode(pdk_clf, train_loader, test_loader, device, mode='pdk', kernel_bank=kernel_bank)
    pdk_final_acc, pdk_eval_time = evaluate_classifier(pdk_clf, test_loader, device, mode='pdk', kernel_bank=kernel_bank)

    with open(os.path.join(SAVE_ROOT, f'{DATASET_NAME.lower()}_raw_log.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_logs[0].keys()))
        writer.writeheader()
        writer.writerows(raw_logs)
    with open(os.path.join(SAVE_ROOT, f'{DATASET_NAME.lower()}_global_log.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(global_logs[0].keys()))
        writer.writeheader()
        writer.writerows(global_logs)
    with open(os.path.join(SAVE_ROOT, f'{DATASET_NAME.lower()}_pdk_log.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(pdk_logs[0].keys()))
        writer.writeheader()
        writer.writerows(pdk_logs)

    summary_rows = [
        {'method': 'Raw', 'best_acc': raw_best_acc, 'final_acc': raw_final_acc, 'eval_time_sec': raw_eval_time},
        {'method': 'Global', 'best_acc': global_best_acc, 'final_acc': global_final_acc, 'eval_time_sec': global_eval_time},
        {'method': 'PDK', 'best_acc': pdk_best_acc, 'final_acc': pdk_final_acc, 'eval_time_sec': pdk_eval_time},
    ]
    with open(os.path.join(SAVE_ROOT, f'{DATASET_NAME.lower()}_classification_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Classification Comparison ===")
    for row in summary_rows:
        print(f"{row['method']}: best_acc={row['best_acc']:.4f}, final_acc={row['final_acc']:.4f}, eval_time={row['eval_time_sec']:.3f}s")


def run_classification():
    run()
