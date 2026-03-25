import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from kernels import (
    get_torch_device,
    apply_global_gaussian_torch,
    kernel_bank_torch,
    apply_zonewise_pdk_soft_torch,
)
from utils import ensure_dir


class ZonePredictor(nn.Module):
    """
    Input: raw, global, |raw-global|   -> 3-zone logits
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


class SmallClassifier(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


def zone_ratio_loss(zone_probs):
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(CFG.CLS_TARGET_ZONE_RATIOS, dtype=zone_probs.dtype, device=zone_probs.device)
    return F.l1_loss(actual, target)


def entropy_loss(zone_probs):
    flat = zone_probs.view(zone_probs.shape[0], zone_probs.shape[1], -1)
    ent = -(flat * torch.log(flat + 1e-8)).sum(dim=2).mean()
    return ent


def build_loaders():
    from torchvision import datasets, transforms

    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    if CFG.CLS_DATASET.upper() == "CIFAR10":
        train_ds = datasets.CIFAR10(root=CFG.CLS_DATA_ROOT, train=True, download=True, transform=tfm)
        test_ds = datasets.CIFAR10(root=CFG.CLS_DATA_ROOT, train=False, download=True, transform=tfm)
        num_classes = 10
    else:
        raise ValueError("Currently starter code supports CIFAR10 only.")

    train_loader = DataLoader(train_ds, batch_size=CFG.CLS_BATCH_SIZE, shuffle=True, num_workers=CFG.CLS_NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=CFG.CLS_BATCH_SIZE, shuffle=False, num_workers=CFG.CLS_NUM_WORKERS)
    return train_loader, test_loader, num_classes


@torch.no_grad()
def evaluate(zone_net, clf, loader, device, kernel_bank):
    zone_net.eval()
    clf.eval()

    total = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        global_x = apply_global_gaussian_torch(x, CFG.GLOBAL_SIGMA_CLASSIFICATION)
        z_in = torch.cat([x, global_x, torch.abs(x - global_x)], dim=1)
        zone_logits = zone_net(z_in)
        zone_probs = F.softmax(zone_logits, dim=1)

        pdk_x = apply_zonewise_pdk_soft_torch(x, zone_probs, kernel_bank)
        pred = clf(pdk_x).argmax(dim=1)

        total += y.numel()
        correct += (pred == y).sum().item()

    return correct / total


def run():
    save_root = os.path.join(CFG.SAVE_ROOT, "classification")
    ensure_dir(save_root)

    device = get_torch_device(CFG.CLS_DEVICE)
    train_loader, test_loader, num_classes = build_loaders()

    zone_net = ZonePredictor(in_ch=9, base=32).to(device)
    clf = SmallClassifier(num_classes=num_classes, in_ch=3).to(device)
    kernel_bank = kernel_bank_torch(CFG.ZONE_KERNEL_SPECS, in_channels=3, device=device)

    optimizer = torch.optim.Adam(
        list(zone_net.parameters()) + list(clf.parameters()),
        lr=CFG.CLS_LR,
        weight_decay=CFG.CLS_WEIGHT_DECAY,
    )

    rows = []

    for epoch in range(1, CFG.CLS_EPOCHS + 1):
        zone_net.train()
        clf.train()

        ce_vals, ratio_vals, ent_vals = [], [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            global_x = apply_global_gaussian_torch(x, CFG.GLOBAL_SIGMA_CLASSIFICATION)
            z_in = torch.cat([x, global_x, torch.abs(x - global_x)], dim=1)

            zone_logits = zone_net(z_in)
            zone_probs = F.softmax(zone_logits, dim=1)

            pdk_x = apply_zonewise_pdk_soft_torch(x, zone_probs, kernel_bank)
            logits = clf(pdk_x)

            ce = F.cross_entropy(logits, y)
            ratio = zone_ratio_loss(zone_probs)
            ent = entropy_loss(zone_probs)

            loss = (
                CFG.CLS_LOSS_W_CE * ce
                + CFG.CLS_LOSS_W_RATIO * ratio
                + CFG.CLS_LOSS_W_ENTROPY * ent
            )
            loss.backward()
            optimizer.step()

            ce_vals.append(float(ce.detach().cpu()))
            ratio_vals.append(float(ratio.detach().cpu()))
            ent_vals.append(float(ent.detach().cpu()))

        acc = evaluate(zone_net, clf, test_loader, device, kernel_bank)
        rows.append({
            "epoch": epoch,
            "ce": float(np.mean(ce_vals)),
            "ratio": float(np.mean(ratio_vals)),
            "entropy": float(np.mean(ent_vals)),
            "test_acc": acc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"CE={np.mean(ce_vals):.5f}, ratio={np.mean(ratio_vals):.5f}, entropy={np.mean(ent_vals):.5f} | "
            f"Test Acc={acc:.4f}"
        )

    csv_path = os.path.join(save_root, "classification_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)