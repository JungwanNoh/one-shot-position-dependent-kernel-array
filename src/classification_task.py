"""
Classification task: downstream proxy for evaluating Raw / Global / PDK preprocessing.

Upgrade notes for this branch:
- Tiny ImageNet uses ShuffleNetV2 x0.5 with ImageNet pretrained weights.
- Input is resized/cropped to 224x224 for Tiny ImageNet.
- ImageNet normalization is applied AFTER Raw / Global / PDK preprocessing.
- Train accuracy is logged every epoch to distinguish optimization failure from generalization failure.
"""

import os
import csv
import time
import copy
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import ShuffleNet_V2_X0_5_Weights

from config import CLASSIFICATION, SEED
from utils import set_seed

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
_C = CLASSIFICATION
DATASET_NAME    = _C['dataset'].upper()
DATA_ROOT       = _C['data_root']
TINY_ROOT       = _C['tiny_imagenet_root']
SAVE_DIR        = _C['save_dir']
BATCH_SIZE      = _C['batch_size']
EPOCHS          = _C['epochs']
LR              = _C['lr']
WEIGHT_DECAY    = _C['weight_decay']
NUM_WORKERS     = _C['num_workers']
GLOBAL_SIGMA    = _C['global_sigma']
INPUT_SIZE      = _C.get('input_size', 224)
USE_IMAGENET_PRETRAINED = _C.get('use_imagenet_pretrained', True)
KERNEL_SIZE     = 3
ZONE_SPECS      = _C['zone_kernel_specs']
RATIO_TARGET    = _C['ratio_target']
LOSS_W_CE       = _C['loss_w_ce']
LOSS_W_RATIO    = _C['loss_w_ratio']
LOSS_W_ENTROPY  = _C['loss_w_entropy']
BACKBONE_NAME   = _C.get('backbone', 'shufflenet')

os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------------------
# Tiny ImageNet validation set (custom loader)
# ----------------------------------------------------------------
class TinyImageNetVal(Dataset):
    def __init__(self, root: str, transform=None):
        wnids = open(os.path.join(root, 'wnids.txt')).read().splitlines()
        cls2idx = {w: i for i, w in enumerate(wnids)}
        val_img_dir = os.path.join(root, 'val', 'images')
        self.samples: List[Tuple[str, int]] = []
        for line in open(os.path.join(root, 'val', 'val_annotations.txt')):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            self.samples.append((os.path.join(val_img_dir, parts[0]), cls2idx[parts[1]]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------
def _build_loaders():
    # Return tensors in [0, 1]. Normalize AFTER preprocessing.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    if DATASET_NAME == 'TINYIMAGENET':
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
        ])
        tr = datasets.ImageFolder(os.path.join(TINY_ROOT, 'train'), transform=train_tfm)
        te = TinyImageNetVal(TINY_ROOT, transform=test_tfm)
        nc = 200
        norm_mean, norm_std = imagenet_mean, imagenet_std
    elif DATASET_NAME == 'CIFAR10':
        train_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tfm = transforms.Compose([transforms.ToTensor()])
        tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tfm)
        te = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tfm)
        nc = 10
        norm_mean, norm_std = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif DATASET_NAME == 'STL10':
        size = max(INPUT_SIZE, 96)
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        tr = datasets.STL10(DATA_ROOT, split='train', download=True, transform=train_tfm)
        te = datasets.STL10(DATA_ROOT, split='test', download=True, transform=test_tfm)
        nc = 10
        norm_mean, norm_std = imagenet_mean, imagenet_std
    else:
        raise ValueError(f'Unsupported dataset: {DATASET_NAME}')

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(te, shuffle=False, **kw),
            nc, norm_mean, norm_std)


# ----------------------------------------------------------------
# Kernel helpers (torch)
# ----------------------------------------------------------------
def _gaussian_np(sigma: float) -> np.ndarray:
    ax = np.arange(-(KERNEL_SIZE // 2), KERNEL_SIZE // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2 + 1e-12))
    return (k / (k.sum() + 1e-12)).astype(np.float32)


def _binomial_np() -> np.ndarray:
    v = np.array([1.], dtype=np.float32)
    for _ in range(KERNEL_SIZE - 1):
        v = np.convolve(v, [1., 1.]).astype(np.float32)
    k = np.outer(v, v)
    return (k / (k.sum() + 1e-12)).astype(np.float32)


def _identity_np() -> np.ndarray:
    k = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
    k[KERNEL_SIZE // 2, KERNEL_SIZE // 2] = 1.
    return k


def _unsharp_np(sigma: float, amount: float) -> np.ndarray:
    return (_identity_np() + amount * (_identity_np() - _gaussian_np(sigma))).astype(np.float32)


def _build_kernel_np(spec: dict) -> np.ndarray:
    family = spec['family']
    if family == 'gaussian':
        return _gaussian_np(float(spec['sigma']))
    if family == 'binomial':
        return _binomial_np()
    if family == 'identity':
        return _identity_np()
    if family == 'unsharp':
        return _unsharp_np(float(spec['sigma']), float(spec['amount']))
    raise ValueError(f'Unknown kernel family: {family}')


def _np_to_depthwise_torch(k_np: np.ndarray, in_channels: int, device: str) -> torch.Tensor:
    t = torch.tensor(k_np, dtype=torch.float32, device=device)
    return t[None, None].expand(in_channels, 1, -1, -1).contiguous()


def _apply_depthwise(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pad = KERNEL_SIZE // 2
    return F.conv2d(x, weight, padding=pad, groups=x.shape[1]).clamp(0., 1.)


def _global_kernel_torch(in_channels: int, device: str) -> torch.Tensor:
    return _np_to_depthwise_torch(_gaussian_np(GLOBAL_SIGMA), in_channels, device)


def _zone_kernel_bank(in_channels: int, device: str) -> List[torch.Tensor]:
    return [_np_to_depthwise_torch(_build_kernel_np(ZONE_SPECS[i]), in_channels, device) for i in [0, 1, 2]]


# ----------------------------------------------------------------
# PDK soft application
# ----------------------------------------------------------------
def _apply_pdk_soft(x: torch.Tensor, zone_probs: torch.Tensor, bank: List[torch.Tensor]) -> torch.Tensor:
    filtered = torch.stack([_apply_depthwise(x, k) for k in bank], dim=1)  # [B,3,C,H,W]
    probs = zone_probs.unsqueeze(2)  # [B,3,1,H,W]
    return (filtered * probs).sum(dim=1).clamp(0., 1.)


# ----------------------------------------------------------------
# ZonePredictor
# ----------------------------------------------------------------
class ZonePredictor(nn.Module):
    def __init__(self, in_ch: int = 9, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------
# Backbone + normalization helpers
# ----------------------------------------------------------------
def _build_classifier(num_classes: int) -> nn.Module:
    if BACKBONE_NAME != 'shufflenet':
        raise ValueError(f'Unsupported backbone: {BACKBONE_NAME}')

    if DATASET_NAME == 'TINYIMAGENET' and USE_IMAGENET_PRETRAINED:
        model = models.shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
    else:
        model = models.shufflenet_v2_x0_5(weights=None)
        if DATASET_NAME == 'CIFAR10':
            model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _normalize_for_backbone(x: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def _model_description(model: nn.Module, num_classes: int) -> str:
    desc = [
        f'Backbone: ShuffleNetV2 x0.5 (pretrained={USE_IMAGENET_PRETRAINED if DATASET_NAME == "TINYIMAGENET" else False})',
        f'Input pipeline: {DATASET_NAME} -> preprocessing (Raw/Global/PDK) -> ImageNet normalize -> classifier',
        'Stem: Conv-BN-ReLU-MaxPool (default stem kept for Tiny ImageNet 224)',
        'Body: ShuffleNetV2 stage2/stage3/stage4 with channel split + shuffle',
        f'Head: Linear -> {num_classes} output logits',
        'Activation: ReLU',
        f'Parameters: {sum(p.numel() for p in model.parameters()):,}',
    ]
    return '\n'.join(desc)


# ----------------------------------------------------------------
# Losses
# ----------------------------------------------------------------
def _ratio_loss(zone_probs: torch.Tensor) -> torch.Tensor:
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(RATIO_TARGET, dtype=zone_probs.dtype, device=zone_probs.device)
    return F.l1_loss(actual, target)


def _entropy_loss(zone_probs: torch.Tensor) -> torch.Tensor:
    flat = zone_probs.view(zone_probs.shape[0], zone_probs.shape[1], -1)
    return -(flat * torch.log(flat + 1e-8)).sum(dim=2).mean()


# ----------------------------------------------------------------
# Train / eval helpers
# ----------------------------------------------------------------
@torch.no_grad()
def _evaluate(clf: nn.Module, loader: DataLoader, device: str, mode: str, norm_mean, norm_std, global_k=None, zone_net=None, bank=None):
    clf.eval()
    if zone_net is not None:
        zone_net.eval()
    correct = total = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mode == 'raw':
            x_in = x
        elif mode == 'global':
            x_in = _apply_depthwise(x, global_k)
        elif mode == 'pdk':
            gx = _apply_depthwise(x, global_k)
            z_in = torch.cat([x, gx, (x - gx).abs()], dim=1)
            probs = F.softmax(zone_net(z_in), dim=1)
            x_in = _apply_pdk_soft(x, probs, bank)
        else:
            raise ValueError(mode)
        x_in = _normalize_for_backbone(x_in, norm_mean, norm_std)
        pred = clf(x_in).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), time.time() - t0


def _train_fixed(clf: nn.Module, loader: DataLoader, test_loader: DataLoader, device: str, mode: str, norm_mean, norm_std, global_k=None):
    opt, sched = _make_opt_sched(clf.parameters())
    best_acc, best_state, logs = 0., None, []
    for ep in range(1, EPOCHS + 1):
        clf.train()
        losses = []
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_in = x if mode == 'raw' else _apply_depthwise(x, global_k)
            x_in = _normalize_for_backbone(x_in, norm_mean, norm_std)
            opt.zero_grad()
            logits = clf(x_in)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
            losses.append(loss.item())
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        sched.step()
        train_acc = correct / max(total, 1)
        acc, _ = _evaluate(clf, test_loader, device, mode, norm_mean, norm_std, global_k)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(clf.state_dict())
        current_lr = opt.param_groups[0]['lr']
        logs.append({'epoch': ep, 'train_ce': float(np.mean(losses)), 'train_acc': train_acc, 'test_acc': acc, 'best_acc': best_acc, 'lr': current_lr})
        print(f'[{mode.upper():6s}] ep{ep:03d} | TrainCE={np.mean(losses):.4f} | TrainAcc={train_acc:.4f} | TestAcc={acc:.4f} | Best={best_acc:.4f} | LR={current_lr:.2e}')
    if best_state is not None:
        clf.load_state_dict(best_state)
    return best_acc, logs


def _train_pdk(clf: nn.Module, zone_net: ZonePredictor, loader: DataLoader, test_loader: DataLoader, device: str, norm_mean, norm_std, global_k, bank):
    params = list(clf.parameters()) + list(zone_net.parameters())
    opt, sched = _make_opt_sched(params)
    best_acc, best_clf, best_zone, logs = 0., None, None, []
    for ep in range(1, EPOCHS + 1):
        clf.train(); zone_net.train()
        ce_l, rat_l, ent_l = [], [], []
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            gx = _apply_depthwise(x, global_k)
            z_in = torch.cat([x, gx, (x - gx).abs()], dim=1)
            opt.zero_grad()
            probs = F.softmax(zone_net(z_in), dim=1)
            x_in = _apply_pdk_soft(x, probs, bank)
            x_in = _normalize_for_backbone(x_in, norm_mean, norm_std)
            logits = clf(x_in)
            ce = F.cross_entropy(logits, y)
            ratio = _ratio_loss(probs)
            ent = _entropy_loss(probs)
            loss = LOSS_W_CE * ce + LOSS_W_RATIO * ratio + LOSS_W_ENTROPY * ent
            loss.backward(); opt.step()
            ce_l.append(ce.item()); rat_l.append(ratio.item()); ent_l.append(ent.item())
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        sched.step()
        train_acc = correct / max(total, 1)
        acc, _ = _evaluate(clf, test_loader, device, 'pdk', norm_mean, norm_std, global_k, zone_net, bank)
        if acc > best_acc:
            best_acc = acc
            best_clf = copy.deepcopy(clf.state_dict())
            best_zone = copy.deepcopy(zone_net.state_dict())
        current_lr = opt.param_groups[0]['lr']
        logs.append({'epoch': ep, 'train_ce': float(np.mean(ce_l)), 'train_acc': train_acc, 'ratio': float(np.mean(rat_l)), 'entropy': float(np.mean(ent_l)), 'test_acc': acc, 'best_acc': best_acc, 'lr': current_lr})
        print(f'[PDK   ] ep{ep:03d} | TrainCE={np.mean(ce_l):.4f} | TrainAcc={train_acc:.4f} | ratio={np.mean(rat_l):.4f} | ent={np.mean(ent_l):.4f} | TestAcc={acc:.4f} | Best={best_acc:.4f} | LR={current_lr:.2e}')
    if best_clf is not None:
        clf.load_state_dict(best_clf)
    if best_zone is not None:
        zone_net.load_state_dict(best_zone)
    return best_acc, logs


def _make_opt_sched(params):
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    return opt, sched


def _save_logs(logs: list, path: str):
    if not logs:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader(); writer.writerows(logs)


# ----------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------
def run():
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, num_classes, norm_mean, norm_std = _build_loaders()

    print(f'\n=== Classification Task [{DATASET_NAME}] on [{device}] ===')
    print(f'Backbone={BACKBONE_NAME}, pretrained={USE_IMAGENET_PRETRAINED}, input_size={INPUT_SIZE}, epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}')
    tmp_model = _build_classifier(num_classes).to(device)
    print('Metric guide: TrainCE=mean train cross-entropy, TrainAcc=train accuracy, TestAcc=test accuracy, Best=best test accuracy so far, LR=current learning rate')
    print('Network structure:')
    print(_model_description(tmp_model, num_classes))
    del tmp_model

    in_ch = 3
    global_k = _global_kernel_torch(in_ch, device)
    bank = _zone_kernel_bank(in_ch, device)

    print('\n--- Training: Raw ---')
    raw_clf = _build_classifier(num_classes).to(device)
    raw_best, raw_logs = _train_fixed(raw_clf, train_loader, test_loader, device, 'raw', norm_mean, norm_std)
    raw_final, raw_t = _evaluate(raw_clf, test_loader, device, 'raw', norm_mean, norm_std)

    print('\n--- Training: Global ---')
    glo_clf = _build_classifier(num_classes).to(device)
    glo_best, glo_logs = _train_fixed(glo_clf, train_loader, test_loader, device, 'global', norm_mean, norm_std, global_k)
    glo_final, glo_t = _evaluate(glo_clf, test_loader, device, 'global', norm_mean, norm_std, global_k)

    print('\n--- Training: PDK ---')
    pdk_clf = _build_classifier(num_classes).to(device)
    zone_net = ZonePredictor(in_ch=in_ch * 3, base=32).to(device)
    pdk_best, pdk_logs = _train_pdk(pdk_clf, zone_net, train_loader, test_loader, device, norm_mean, norm_std, global_k, bank)
    pdk_final, pdk_t = _evaluate(pdk_clf, test_loader, device, 'pdk', norm_mean, norm_std, global_k, zone_net, bank)

    ds = DATASET_NAME.lower()
    _save_logs(raw_logs, os.path.join(SAVE_DIR, f'{ds}_raw_log.csv'))
    _save_logs(glo_logs, os.path.join(SAVE_DIR, f'{ds}_global_log.csv'))
    _save_logs(pdk_logs, os.path.join(SAVE_DIR, f'{ds}_pdk_log.csv'))

    summary = [
        {'method': 'Raw', 'best_acc': raw_best, 'final_acc': raw_final, 'eval_sec': raw_t},
        {'method': 'Global', 'best_acc': glo_best, 'final_acc': glo_final, 'eval_sec': glo_t},
        {'method': 'PDK', 'best_acc': pdk_best, 'final_acc': pdk_final, 'eval_sec': pdk_t},
    ]
    with open(os.path.join(SAVE_DIR, f'{ds}_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    print('\n=== Classification Summary ===')
    for row in summary:
        print(f"  {row['method']:6s} | best={row['best_acc']:.4f} final={row['final_acc']:.4f} eval={row['eval_sec']:.2f}s")
    print()
