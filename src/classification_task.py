import os
import csv
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import COMMON, CLASSIFICATION, DEVICE, SEED
from utils import ensure_dir, set_seed
from kernels import apply_global_torch, get_zone_kernel_bank_torch, apply_pdk_soft_torch, apply_pdk_hard_torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ZoneHeatmapHead(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ConvBlock(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec2 = ConvBlock(base * 2, base)
        self.heatmap_head = nn.Conv2d(base, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.radius_head = nn.Sequential(nn.Flatten(), nn.Linear(base * 4, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        logits = self.heatmap_head(d2)
        probs, cx, cy = self.softargmax(logits, CLASSIFICATION["heatmap_softmax_temp"])

        raw = self.radius_head(self.gap(b))
        r1 = CLASSIFICATION["r1_min"] + (CLASSIFICATION["r1_max"] - CLASSIFICATION["r1_min"]) * torch.sigmoid(raw[:, 0])
        dr = CLASSIFICATION["dr_min"] + (CLASSIFICATION["dr_max"] - CLASSIFICATION["dr_min"]) * torch.sigmoid(raw[:, 1])
        r2 = r1 + dr
        return {"heatmap_probs": probs, "cx": cx, "cy": cy, "r1": r1, "r2": r2}

    @staticmethod
    def softargmax(logits, temp):
        b, _, h, w = logits.shape
        flat = (logits / temp).view(b, -1)
        probs = F.softmax(flat, dim=1).view(b, 1, h, w)
        yy = torch.linspace(0.0, 1.0, h, device=logits.device).view(1, 1, h, 1)
        xx = torch.linspace(0.0, 1.0, w, device=logits.device).view(1, 1, 1, w)
        cx = (probs * xx).sum(dim=(1, 2, 3))
        cy = (probs * yy).sum(dim=(1, 2, 3))
        return probs, cx, cy


def params_to_soft_zones(params, h, w, device):
    yy = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, w)
    yy = yy.expand(params["cx"].shape[0], h, w)
    xx = xx.expand(params["cx"].shape[0], h, w)
    cx = params["cx"].view(-1, 1, 1)
    cy = params["cy"].view(-1, 1, 1)
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-8).unsqueeze(1)
    r1 = params["r1"].view(-1, 1, 1, 1)
    r2 = params["r2"].view(-1, 1, 1, 1)
    temp = CLASSIFICATION["zone_soft_temp"]
    attn = torch.sigmoid((r1 - dist) / temp)
    around = torch.sigmoid((dist - r2) / temp)
    inter = (1.0 - attn - around).clamp(min=1e-6)
    probs = torch.cat([attn, inter, around], dim=1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    return probs


def params_to_hard_zones(params, h, w, device):
    yy = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=device).view(1, 1, w)
    yy = yy.expand(params["cx"].shape[0], h, w)
    xx = xx.expand(params["cx"].shape[0], h, w)
    cx = params["cx"].view(-1, 1, 1)
    cy = params["cy"].view(-1, 1, 1)
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-8)
    z = torch.full_like(dist, 2, dtype=torch.long)
    z = torch.where(dist <= params["r2"].view(-1, 1, 1), torch.ones_like(z), z)
    z = torch.where(dist <= params["r1"].view(-1, 1, 1), torch.zeros_like(z), z)
    return z


class SmallClassifier(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


class RawModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clf = SmallClassifier(3, num_classes)

    def forward(self, x):
        return self.clf(x), {}


class GlobalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clf = SmallClassifier(3, num_classes)
        self.global_spec = COMMON["global_specs"]["classification"]
        self.kernel_size = COMMON["kernel_size"]

    def forward(self, x):
        xg = apply_global_torch(x, self.global_spec, self.kernel_size)
        return self.clf(xg), {}


class PDKModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.zone_head = ZoneHeatmapHead(in_ch=3, base=32)
        self.clf = SmallClassifier(3, num_classes)
        self.kernel_bank = get_zone_kernel_bank_torch(COMMON["zone_kernel_specs"], COMMON["kernel_size"], device)

    def forward(self, x):
        params = self.zone_head(x)
        if self.training:
            zone_probs = params_to_soft_zones(params, x.shape[-2], x.shape[-1], x.device)
            xp = apply_pdk_soft_torch(x, zone_probs, self.kernel_bank)
            logits = self.clf(xp)
            aux = {"zone_probs": zone_probs, **params}
        else:
            zone_map = params_to_hard_zones(params, x.shape[-2], x.shape[-1], x.device)
            xp = apply_pdk_hard_torch(x, zone_map, self.kernel_bank)
            logits = self.clf(xp)
            aux = {"zone_map": zone_map, **params}
        return logits, aux


def zone_ratio_loss(zone_probs):
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(CLASSIFICATION["ratio_target"], device=zone_probs.device, dtype=zone_probs.dtype)
    return F.l1_loss(actual, target)


def entropy_loss(heatmap_probs):
    b, _, h, w = heatmap_probs.shape
    flat = heatmap_probs.view(b, -1)
    ent = -(flat * torch.log(flat + 1e-8)).sum(dim=1)
    ent = ent / np.log(h * w)
    return ent.mean()


def build_dataloaders():
    from torchvision import datasets, transforms

    if CLASSIFICATION["dataset"].upper() == "CIFAR10":
        tf = transforms.Compose([transforms.ToTensor()])
        train_all = datasets.CIFAR10(root=CLASSIFICATION["data_root"], train=True, download=True, transform=tf)
        test_ds = datasets.CIFAR10(root=CLASSIFICATION["data_root"], train=False, download=True, transform=tf)
        num_classes = 10
    elif CLASSIFICATION["dataset"].upper() == "STL10":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((96, 96))])
        train_all = datasets.STL10(root=CLASSIFICATION["data_root"], split="train", download=True, transform=tf)
        test_ds = datasets.STL10(root=CLASSIFICATION["data_root"], split="test", download=True, transform=tf)
        num_classes = 10
    else:
        raise ValueError(CLASSIFICATION["dataset"])

    val_len = max(1, int(0.1 * len(train_all)))
    train_len = len(train_all) - val_len
    train_ds, val_ds = random_split(train_all, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=True, num_workers=CLASSIFICATION["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=False, num_workers=CLASSIFICATION["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=False, num_workers=CLASSIFICATION["num_workers"])
    return train_loader, val_loader, test_loader, num_classes


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            total += y.numel()
            correct += (pred == y).sum().item()
    return 100.0 * correct / total


def train_one_model(mode: str, device: str, num_classes: int, train_loader, val_loader, test_loader):
    if mode == "raw":
        model = RawModel(num_classes).to(device)
    elif mode == "global":
        model = GlobalModel(num_classes).to(device)
    elif mode == "pdk":
        model = PDKModel(num_classes, device).to(device)
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam(model.parameters(), lr=CLASSIFICATION["lr"], weight_decay=CLASSIFICATION["weight_decay"])
    best_state = None
    best_val = -1.0

    for epoch in range(1, CLASSIFICATION["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, aux = model(x)
            ce = F.cross_entropy(logits, y)
            loss = CLASSIFICATION["loss_w_ce"] * ce
            if mode == "pdk":
                loss = loss + CLASSIFICATION["loss_w_ratio"] * zone_ratio_loss(aux["zone_probs"])
                loss = loss + CLASSIFICATION["loss_w_entropy"] * entropy_loss(aux["heatmap_probs"])
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * x.shape[0]
            total_items += x.shape[0]

        val_acc = evaluate(model, val_loader, device)
        train_loss = total_loss / max(total_items, 1)
        print(f"[{mode}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.2f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    return model, best_val, test_acc


def run_classification():
    set_seed(SEED)
    device = "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    save_dir = CLASSIFICATION["save_dir"]
    ensure_dir(save_dir)

    train_loader, val_loader, test_loader, num_classes = build_dataloaders()
    rows = []
    for mode in ["raw", "global", "pdk"]:
        _, best_val, test_acc = train_one_model(mode, device, num_classes, train_loader, val_loader, test_loader)
        rows.append({"mode": mode, "best_val_acc": best_val, "test_acc": test_acc})

    csv_path = os.path.join(save_dir, "classification_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    print("\n=== Classification Summary ===")
    for row in rows:
        print(row)
