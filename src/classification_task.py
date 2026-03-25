import os
import csv
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from config import COMMON, CLASSIFICATION, DEVICE, SEED
from utils import ensure_dir, set_seed
from kernels import apply_global_torch, get_zone_kernel_bank_torch, apply_pdk_soft_torch


class TinyImageNetValDataset(Dataset):
    """
    Standard tiny-imagenet-200 val loader without requiring folder reshuffling.
    Expects:
      root/
        train/<class>/images/*.JPEG
        wnids.txt
        val/images/*.JPEG
        val/val_annotations.txt
    """
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        wnids = (self.root / "wnids.txt").read_text().strip().splitlines()
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

        ann_path = self.root / "val" / "val_annotations.txt"
        img_dir = self.root / "val" / "images"
        self.samples = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split("\t")
                fname, wnid = toks[0], toks[1]
                self.samples.append((img_dir / fname, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


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
    def __init__(self, in_ch=9, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)


class RawModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clf = ResNet18Classifier(num_classes)

    def forward(self, x):
        return self.clf(x), {}


class GlobalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clf = ResNet18Classifier(num_classes)
        self.global_spec = COMMON["global_specs"]["classification"]
        self.kernel_size = COMMON["kernel_size"]

    def forward(self, x):
        xg = apply_global_torch(x, self.global_spec, self.kernel_size)
        return self.clf(xg), {}


class PDKModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.zone_head = ZoneHeatmapHead(in_ch=9, base=32)
        self.clf = ResNet18Classifier(num_classes)
        self.kernel_bank = get_zone_kernel_bank_torch(COMMON["zone_kernel_specs"], COMMON["kernel_size"], device)
        self.global_spec = COMMON["global_specs"]["classification"]
        self.kernel_size = COMMON["kernel_size"]

    def forward(self, x):
        xg = apply_global_torch(x, self.global_spec, self.kernel_size)
        z_in = torch.cat([x, xg, torch.abs(x - xg)], dim=1)
        zone_logits = self.zone_head(z_in)
        zone_probs = F.softmax(zone_logits, dim=1)
        xp = apply_pdk_soft_torch(x, zone_probs, self.kernel_bank)
        logits = self.clf(xp)
        aux = {"zone_probs": zone_probs}
        return logits, aux


def zone_ratio_loss(zone_probs):
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(CLASSIFICATION["ratio_target"], device=zone_probs.device, dtype=zone_probs.dtype)
    return F.l1_loss(actual, target)


def entropy_loss(zone_probs):
    flat = zone_probs.view(zone_probs.shape[0], zone_probs.shape[1], -1)
    ent = -(flat * torch.log(flat + 1e-8)).sum(dim=2).mean()
    return ent


def build_transforms(dataset_name: str):
    from torchvision import transforms

    use_ra = CLASSIFICATION["use_randaugment"]
    ra = transforms.RandAugment(
        num_ops=CLASSIFICATION["randaugment_num_ops"],
        magnitude=CLASSIFICATION["randaugment_magnitude"],
    ) if use_ra else None

    name = dataset_name.upper()
    if name == "CIFAR10":
        train_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if ra is not None:
            train_ops.append(ra)
        train_ops.append(transforms.ToTensor())
        test_ops = [transforms.ToTensor()]
    elif name == "STL10":
        train_ops = [transforms.RandomCrop(96, padding=12), transforms.RandomHorizontalFlip()]
        if ra is not None:
            train_ops.append(ra)
        train_ops.append(transforms.ToTensor())
        test_ops = [transforms.ToTensor()]
    elif name == "TINYIMAGENET":
        train_ops = [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
        if ra is not None:
            train_ops.append(ra)
        train_ops.append(transforms.ToTensor())
        test_ops = [transforms.ToTensor()]
    else:
        raise ValueError(dataset_name)

    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def build_dataloaders():
    from torchvision import datasets

    dataset_name = CLASSIFICATION["dataset"].upper()
    tf_train, tf_test = build_transforms(dataset_name)

    if dataset_name == "CIFAR10":
        train_all = datasets.CIFAR10(root=CLASSIFICATION["data_root"], train=True, download=True, transform=tf_train)
        test_ds = datasets.CIFAR10(root=CLASSIFICATION["data_root"], train=False, download=True, transform=tf_test)
        num_classes = 10
    elif dataset_name == "STL10":
        train_all = datasets.STL10(root=CLASSIFICATION["data_root"], split="train", download=True, transform=tf_train)
        test_ds = datasets.STL10(root=CLASSIFICATION["data_root"], split="test", download=True, transform=tf_test)
        num_classes = 10
    elif dataset_name == "TINYIMAGENET":
        root = CLASSIFICATION["tiny_imagenet_root"]
        train_all = datasets.ImageFolder(root=os.path.join(root, "train"), transform=tf_train)
        test_ds = TinyImageNetValDataset(root=root, transform=tf_test)
        num_classes = 200
    else:
        raise ValueError(CLASSIFICATION["dataset"])

    val_len = max(1, int(0.1 * len(train_all)))
    train_len = len(train_all) - val_len
    train_ds, val_ds = random_split(train_all, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=True, num_workers=CLASSIFICATION["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=False, num_workers=CLASSIFICATION["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CLASSIFICATION["batch_size"], shuffle=False, num_workers=CLASSIFICATION["num_workers"], pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLASSIFICATION["epochs"])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    start = time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return 100.0 * correct / total, time.time() - start


def train_one_model(mode: str, device: str, num_classes: int, train_loader, val_loader, test_loader):
    if mode == "raw":
        model = RawModel(num_classes).to(device)
    elif mode == "global":
        model = GlobalModel(num_classes).to(device)
    elif mode == "pdk":
        model = PDKModel(num_classes, device).to(device)
    else:
        raise ValueError(mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CLASSIFICATION["lr"], weight_decay=CLASSIFICATION["weight_decay"])
    scheduler = build_scheduler(optimizer)

    best_state = None
    best_val = -1.0
    logs = []

    for epoch in range(1, CLASSIFICATION["epochs"] + 1):
        model.train()
        ce_vals, ratio_vals, ent_vals = [], [], []

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, aux = model(x)
            ce = F.cross_entropy(logits, y)
            ratio = torch.tensor(0.0, device=device)
            ent = torch.tensor(0.0, device=device)
            if mode == "pdk":
                ratio = zone_ratio_loss(aux["zone_probs"])
                ent = entropy_loss(aux["zone_probs"])
            loss = ce + CLASSIFICATION["loss_w_ratio"] * ratio + CLASSIFICATION["loss_w_entropy"] * ent
            loss.backward()
            optimizer.step()
            ce_vals.append(float(ce.detach().cpu()))
            ratio_vals.append(float(ratio.detach().cpu()))
            ent_vals.append(float(ent.detach().cpu()))

        scheduler.step()
        val_acc, _ = evaluate(model, val_loader, device)
        test_acc, _ = evaluate(model, test_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

        logs.append({
            "epoch": epoch,
            "ce": float(np.mean(ce_vals)),
            "ratio": float(np.mean(ratio_vals)),
            "entropy": float(np.mean(ent_vals)),
            "val_acc": val_acc,
            "test_acc": test_acc,
        })

        print(
            f"[{mode.upper()}] Epoch {epoch:03d} | CE={np.mean(ce_vals):.5f}, ratio={np.mean(ratio_vals):.5f}, entropy={np.mean(ent_vals):.5f} | Val={val_acc:.3f} | Test={test_acc:.3f}"
        )

    model.load_state_dict(best_state)
    final_test_acc, eval_time = evaluate(model, test_loader, device)
    return final_test_acc, eval_time, logs


def run_classification():
    save_root = CLASSIFICATION["save_dir"]
    ensure_dir(save_root)
    set_seed(SEED)

    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    train_loader, val_loader, test_loader, num_classes = build_dataloaders()

    summary_rows = []
    all_logs = {}

    for mode in ["raw", "global", "pdk"]:
        test_acc, eval_time, logs = train_one_model(mode, device, num_classes, train_loader, val_loader, test_loader)
        all_logs[mode] = logs
        summary_rows.append({
            "method": mode.capitalize() if mode != "pdk" else "PDK",
            "dataset": CLASSIFICATION["dataset"],
            "test_acc": test_acc,
            "eval_time_sec": eval_time,
        })

    with open(os.path.join(save_root, f"classification_summary_{CLASSIFICATION['dataset'].lower()}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save per-method learning logs
    for mode, logs in all_logs.items():
        with open(os.path.join(save_root, f"{mode}_log_{CLASSIFICATION['dataset'].lower()}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)

    print("\n=== Classification Comparison ===")
    for row in summary_rows:
        print(f"{row['method']}: test_acc={row['test_acc']:.3f}, eval_time={row['eval_time_sec']:.3f}s")
