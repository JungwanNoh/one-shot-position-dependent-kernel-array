import os
import csv
import time
import copy
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms, models

from config import CLASSIFICATION


# ============================================================
# Config helpers
# ============================================================

DATASET_NAME = CLASSIFICATION.get("dataset", "CIFAR10").upper()
DATA_ROOT = CLASSIFICATION.get("data_root", "../dat/torchvision")
TINY_IMAGENET_ROOT = CLASSIFICATION.get("tiny_imagenet_root", "../dat/tiny-imagenet-200")

DEVICE = CLASSIFICATION.get("device", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = CLASSIFICATION.get("batch_size", 128)
EPOCHS = CLASSIFICATION.get("epochs", 20)
LR = CLASSIFICATION.get("lr", 1e-3)
WEIGHT_DECAY = CLASSIFICATION.get("weight_decay", 1e-5)
NUM_WORKERS = CLASSIFICATION.get("num_workers", 0)

# Global / PDK settings
GLOBAL_SIGMA = CLASSIFICATION.get("global_sigma", 0.7)
KERNEL_SIZE = CLASSIFICATION.get("kernel_size", 3)

# PDK kernel specs
ZONE_KERNEL_SPECS = CLASSIFICATION.get(
    "zone_kernel_specs",
    {
        0: {"family": "unsharp", "sigma": 0.5, "amount": 0.12},  # attention
        1: {"family": "binomial"},                                # intermediate
        2: {"family": "gaussian", "sigma": 0.9},                  # around
    },
)

# Zone regularization
TARGET_ZONE_RATIOS = CLASSIFICATION.get("target_zone_ratios", (0.15, 0.25, 0.60))
LOSS_W_CE = CLASSIFICATION.get("loss_w_ce", 1.0)
LOSS_W_RATIO = CLASSIFICATION.get("loss_w_ratio", 0.01)
LOSS_W_ENTROPY = CLASSIFICATION.get("loss_w_entropy", 0.002)

SAVE_ROOT = CLASSIFICATION.get("save_root", "../res/classification_compare")
os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str) -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Tiny ImageNet validation loader
# ============================================================

class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        wnids_path = os.path.join(root, "wnids.txt")
        val_annot_path = os.path.join(root, "val", "val_annotations.txt")
        val_img_dir = os.path.join(root, "val", "images")

        with open(wnids_path, "r", encoding="utf-8") as f:
            wnids = [line.strip() for line in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        self.samples: List[Tuple[str, int]] = []
        with open(val_annot_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                img_path = os.path.join(val_img_dir, img_name)
                label = self.class_to_idx[wnid]
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============================================================
# Data
# ============================================================

def build_transforms(dataset_name: str):
    if dataset_name == "CIFAR10":
        image_size = 32
    elif dataset_name == "STL10":
        image_size = 96
    elif dataset_name == "TINYIMAGENET":
        image_size = 64
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_tfm = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_tfm, test_tfm


def build_loaders():
    train_tfm, test_tfm = build_transforms(DATASET_NAME)

    if DATASET_NAME == "CIFAR10":
        train_ds = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_tfm)
        test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tfm)
        num_classes = 10

    elif DATASET_NAME == "STL10":
        train_ds = datasets.STL10(root=DATA_ROOT, split="train", download=True, transform=train_tfm)
        test_ds = datasets.STL10(root=DATA_ROOT, split="test", download=True, transform=test_tfm)
        num_classes = 10

    elif DATASET_NAME == "TINYIMAGENET":
        train_root = os.path.join(TINY_IMAGENET_ROOT, "train")
        train_ds = datasets.ImageFolder(root=train_root, transform=train_tfm)
        test_ds = TinyImageNetValDataset(root=TINY_IMAGENET_ROOT, transform=test_tfm)
        num_classes = 200

    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, test_loader, num_classes


# ============================================================
# Kernels
# ============================================================

def identity_kernel_np(kernel_size: int) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    k[c, c] = 1.0
    return k


def gaussian_kernel_np(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-12))
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def binomial_kernel_np(kernel_size: int) -> np.ndarray:
    order = kernel_size - 1
    vec = np.array([1.0], dtype=np.float32)
    for _ in range(order):
        vec = np.convolve(vec, np.array([1.0, 1.0], dtype=np.float32)).astype(np.float32)
    kernel = np.outer(vec, vec)
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def unsharp_kernel_np(kernel_size: int, sigma: float, amount: float) -> np.ndarray:
    delta = identity_kernel_np(kernel_size)
    blur = gaussian_kernel_np(kernel_size, sigma)
    kernel = delta + amount * (delta - blur)
    return kernel.astype(np.float32)


def build_kernel_np(spec: dict, kernel_size: int) -> np.ndarray:
    family = spec["family"]
    if family == "gaussian":
        return gaussian_kernel_np(kernel_size, float(spec["sigma"]))
    if family == "binomial":
        return binomial_kernel_np(kernel_size)
    if family == "unsharp":
        return unsharp_kernel_np(kernel_size, float(spec["sigma"]), float(spec["amount"]))
    raise ValueError(f"Unknown kernel family: {family}")


def kernel_bank_torch(zone_specs: dict, in_channels: int, device: str):
    bank = []
    for zone_id in [0, 1, 2]:
        k = build_kernel_np(zone_specs[zone_id], KERNEL_SIZE)
        k = torch.tensor(k, dtype=torch.float32, device=device)[None, None, :, :]
        k = k.repeat(in_channels, 1, 1, 1)  # depthwise
        bank.append(k)
    return bank


def apply_global_gaussian_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    c = x.shape[1]
    k = gaussian_kernel_np(KERNEL_SIZE, sigma)
    k = torch.tensor(k, dtype=torch.float32, device=x.device)[None, None, :, :].repeat(c, 1, 1, 1)
    pad = KERNEL_SIZE // 2
    out = F.conv2d(x, k, padding=pad, groups=c)
    return out.clamp(0.0, 1.0)


def apply_zonewise_pdk_soft_torch(x: torch.Tensor, zone_probs: torch.Tensor, bank) -> torch.Tensor:
    pad = KERNEL_SIZE // 2

    filtered = []
    for k in bank:
        filtered.append(F.conv2d(x, k, padding=pad, groups=x.shape[1]))
    filtered = torch.stack(filtered, dim=1)  # [B,3,C,H,W]

    zone_probs = zone_probs.unsqueeze(2)      # [B,3,1,H,W]
    out = (filtered * zone_probs).sum(dim=1)
    return out.clamp(0.0, 1.0)


# ============================================================
# Models
# ============================================================

class ZonePredictor(nn.Module):
    """
    Input: raw, global, |raw-global| -> 3-zone logits
    """
    def __init__(self, in_ch=9, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_classifier(num_classes: int, dataset_name: str):
    model = models.shufflenet_v2_x0_5(weights=None)

    # Small-image friendly stem
    if dataset_name.upper() in ["CIFAR10", "STL10", "TINYIMAGENET"]:
        model.conv1[0] = nn.Conv2d(
            3, 24, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# Losses
# ============================================================

def zone_ratio_loss(zone_probs):
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(TARGET_ZONE_RATIOS, dtype=zone_probs.dtype, device=zone_probs.device)
    return F.l1_loss(actual, target)


def entropy_loss(zone_probs):
    flat = zone_probs.view(zone_probs.shape[0], zone_probs.shape[1], -1)
    ent = -(flat * torch.log(flat + 1e-8)).sum(dim=2).mean()
    return ent


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_classifier(
    clf,
    loader,
    device,
    preprocess_mode="raw",
    zone_net=None,
    kernel_bank=None,
):
    clf.eval()
    if zone_net is not None:
        zone_net.eval()

    total = 0
    correct = 0

    t0 = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if preprocess_mode == "raw":
            x_in = x

        elif preprocess_mode == "global":
            x_in = apply_global_gaussian_torch(x, GLOBAL_SIGMA)

        elif preprocess_mode == "pdk":
            global_x = apply_global_gaussian_torch(x, GLOBAL_SIGMA)
            z_in = torch.cat([x, global_x, torch.abs(x - global_x)], dim=1)
            zone_logits = zone_net(z_in)
            zone_probs = F.softmax(zone_logits, dim=1)
            x_in = apply_zonewise_pdk_soft_torch(x, zone_probs, kernel_bank)

        else:
            raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")

        pred = clf(x_in).argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()

    elapsed = time.time() - t0
    return correct / total, elapsed


# ============================================================
# Train loops
# ============================================================

def build_optimizer_and_scheduler(model_params):
    optimizer = torch.optim.AdamW(
        model_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )
    return optimizer, scheduler


def train_raw_or_global(clf, train_loader, test_loader, device, preprocess_mode="raw"):
    optimizer, scheduler = build_optimizer_and_scheduler(clf.parameters())

    best_acc = 0.0
    best_state = None
    logs = []

    for epoch in range(1, EPOCHS + 1):
        clf.train()
        ce_vals = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if preprocess_mode == "raw":
                x_in = x
            elif preprocess_mode == "global":
                x_in = apply_global_gaussian_torch(x, GLOBAL_SIGMA)
            else:
                raise ValueError("preprocess_mode must be raw or global")

            optimizer.zero_grad()
            logits = clf(x_in)
            ce = F.cross_entropy(logits, y)
            ce.backward()
            optimizer.step()

            ce_vals.append(float(ce.detach().cpu()))

        scheduler.step()

        acc, _ = evaluate_classifier(clf, test_loader, device, preprocess_mode=preprocess_mode)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(clf.state_dict())

        logs.append({
            "epoch": epoch,
            "ce": float(np.mean(ce_vals)),
            "test_acc": acc,
            "best_acc": best_acc,
        })

        print(
            f"[{preprocess_mode.upper()}] Epoch {epoch:03d} | "
            f"CE={np.mean(ce_vals):.5f} | Test Acc={acc:.4f} | Best={best_acc:.4f}"
        )

    if best_state is not None:
        clf.load_state_dict(best_state)

    return best_acc, logs


def train_pdk(clf, zone_net, train_loader, test_loader, device, kernel_bank):
    optimizer, scheduler = build_optimizer_and_scheduler(
        list(clf.parameters()) + list(zone_net.parameters())
    )

    best_acc = 0.0
    best_clf_state = None
    best_zone_state = None
    logs = []

    for epoch in range(1, EPOCHS + 1):
        clf.train()
        zone_net.train()

        ce_vals, ratio_vals, ent_vals = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            global_x = apply_global_gaussian_torch(x, GLOBAL_SIGMA)
            z_in = torch.cat([x, global_x, torch.abs(x - global_x)], dim=1)

            optimizer.zero_grad()

            zone_logits = zone_net(z_in)
            zone_probs = F.softmax(zone_logits, dim=1)
            pdk_x = apply_zonewise_pdk_soft_torch(x, zone_probs, kernel_bank)

            logits = clf(pdk_x)

            ce = F.cross_entropy(logits, y)
            ratio = zone_ratio_loss(zone_probs)
            ent = entropy_loss(zone_probs)

            loss = (
                LOSS_W_CE * ce
                + LOSS_W_RATIO * ratio
                + LOSS_W_ENTROPY * ent
            )
            loss.backward()
            optimizer.step()

            ce_vals.append(float(ce.detach().cpu()))
            ratio_vals.append(float(ratio.detach().cpu()))
            ent_vals.append(float(ent.detach().cpu()))

        scheduler.step()

        acc, _ = evaluate_classifier(
            clf, test_loader, device,
            preprocess_mode="pdk",
            zone_net=zone_net,
            kernel_bank=kernel_bank,
        )

        if acc > best_acc:
            best_acc = acc
            best_clf_state = copy.deepcopy(clf.state_dict())
            best_zone_state = copy.deepcopy(zone_net.state_dict())

        logs.append({
            "epoch": epoch,
            "ce": float(np.mean(ce_vals)),
            "ratio": float(np.mean(ratio_vals)),
            "entropy": float(np.mean(ent_vals)),
            "test_acc": acc,
            "best_acc": best_acc,
        })

        print(
            f"[PDK] Epoch {epoch:03d} | "
            f"CE={np.mean(ce_vals):.5f}, ratio={np.mean(ratio_vals):.5f}, entropy={np.mean(ent_vals):.5f} | "
            f"Test Acc={acc:.4f} | Best={best_acc:.4f}"
        )

    if best_clf_state is not None:
        clf.load_state_dict(best_clf_state)
    if best_zone_state is not None:
        zone_net.load_state_dict(best_zone_state)

    return best_acc, logs


# ============================================================
# Main
# ============================================================

def run():
    device = get_device(DEVICE)
    train_loader, test_loader, num_classes = build_loaders()

    # -----------------------------
    # RAW
    # -----------------------------
    raw_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    raw_best_acc, raw_logs = train_raw_or_global(
        raw_clf, train_loader, test_loader, device, preprocess_mode="raw"
    )
    raw_final_acc, raw_eval_time = evaluate_classifier(
        raw_clf, test_loader, device, preprocess_mode="raw"
    )

    # -----------------------------
    # GLOBAL
    # -----------------------------
    global_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    global_best_acc, global_logs = train_raw_or_global(
        global_clf, train_loader, test_loader, device, preprocess_mode="global"
    )
    global_final_acc, global_eval_time = evaluate_classifier(
        global_clf, test_loader, device, preprocess_mode="global"
    )

    # -----------------------------
    # PDK
    # -----------------------------
    pdk_clf = build_classifier(num_classes, DATASET_NAME).to(device)
    zone_net = ZonePredictor(in_ch=9, base=32).to(device)
    kernel_bank = kernel_bank_torch(ZONE_KERNEL_SPECS, in_channels=3, device=device)

    pdk_best_acc, pdk_logs = train_pdk(
        pdk_clf, zone_net, train_loader, test_loader, device, kernel_bank
    )
    pdk_final_acc, pdk_eval_time = evaluate_classifier(
        pdk_clf, test_loader, device,
        preprocess_mode="pdk",
        zone_net=zone_net,
        kernel_bank=kernel_bank,
    )

    # -----------------------------
    # Save logs
    # -----------------------------
    with open(os.path.join(SAVE_ROOT, f"{DATASET_NAME.lower()}_raw_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_logs[0].keys()))
        writer.writeheader()
        writer.writerows(raw_logs)

    with open(os.path.join(SAVE_ROOT, f"{DATASET_NAME.lower()}_global_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_logs[0].keys()))
        writer.writeheader()
        writer.writerows(global_logs)

    with open(os.path.join(SAVE_ROOT, f"{DATASET_NAME.lower()}_pdk_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pdk_logs[0].keys()))
        writer.writeheader()
        writer.writerows(pdk_logs)

    summary_rows = [
        {
            "method": "Raw",
            "best_acc": raw_best_acc,
            "final_acc": raw_final_acc,
            "eval_time_sec": raw_eval_time,
        },
        {
            "method": "Global",
            "best_acc": global_best_acc,
            "final_acc": global_final_acc,
            "eval_time_sec": global_eval_time,
        },
        {
            "method": "PDK",
            "best_acc": pdk_best_acc,
            "final_acc": pdk_final_acc,
            "eval_time_sec": pdk_eval_time,
        },
    ]

    with open(os.path.join(SAVE_ROOT, f"{DATASET_NAME.lower()}_classification_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Classification Comparison ===")
    for row in summary_rows:
        print(
            f"{row['method']}: "
            f"best_acc={row['best_acc']:.4f}, "
            f"final_acc={row['final_acc']:.4f}, "
            f"eval_time={row['eval_time_sec']:.3f}s"
        )