import numpy as np
from scipy.ndimage import convolve

import torch
import torch.nn.functional as F


def identity_kernel(kernel_size: int) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    k[c, c] = 1.0
    return k


def gaussian_kernel_np(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2 + 1e-12))
    k /= k.sum() + 1e-12
    return k.astype(np.float32)


def binomial_kernel_np(kernel_size: int) -> np.ndarray:
    order = kernel_size - 1
    vec = np.array([1.0], dtype=np.float32)
    for _ in range(order):
        vec = np.convolve(vec, np.array([1.0, 1.0], dtype=np.float32)).astype(np.float32)
    k = np.outer(vec, vec)
    k /= k.sum() + 1e-12
    return k.astype(np.float32)


def unsharp_kernel_np(kernel_size: int, sigma: float, amount: float) -> np.ndarray:
    delta = identity_kernel(kernel_size)
    blur = gaussian_kernel_np(kernel_size, sigma)
    k = delta + amount * (delta - blur)
    return k.astype(np.float32)


def build_kernel_np(spec: dict, kernel_size: int) -> np.ndarray:
    family = spec["family"]
    if family == "identity":
        return identity_kernel(kernel_size)
    if family == "gaussian":
        return gaussian_kernel_np(kernel_size, float(spec["sigma"]))
    if family == "binomial":
        return binomial_kernel_np(kernel_size)
    if family == "unsharp":
        return unsharp_kernel_np(kernel_size, float(spec["sigma"]), float(spec["amount"]))
    raise ValueError(f"Unknown kernel family: {family}")


def apply_global_np(img: np.ndarray, spec: dict, kernel_size: int) -> np.ndarray:
    k = build_kernel_np(spec, kernel_size)
    out = convolve(img, k, mode="reflect")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_pdk_np(img: np.ndarray, zone_map: np.ndarray, zone_specs: dict, kernel_size: int) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float32)
    bank = {}
    for zid in [0, 1, 2]:
        k = build_kernel_np(zone_specs[zid], kernel_size)
        bank[zid] = convolve(img, k, mode="reflect").astype(np.float32)
    for zid in [0, 1, 2]:
        mask = zone_map == zid
        out[mask] = bank[zid][mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _depthwise_conv(img: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    # img: [B,C,H,W], kernel_2d: [K,K]
    b, c, _, _ = img.shape
    k = kernel_2d.shape[-1]
    pad = k // 2
    weight = kernel_2d.view(1, 1, k, k).repeat(c, 1, 1, 1)
    return F.conv2d(img, weight, padding=pad, groups=c)


def build_kernel_torch(spec: dict, kernel_size: int, device: str) -> torch.Tensor:
    return torch.tensor(build_kernel_np(spec, kernel_size), dtype=torch.float32, device=device)


def apply_global_torch(img: torch.Tensor, spec: dict, kernel_size: int) -> torch.Tensor:
    k = build_kernel_torch(spec, kernel_size, img.device)
    out = _depthwise_conv(img, k)
    return out.clamp(0.0, 1.0)


def get_zone_kernel_bank_torch(zone_specs: dict, kernel_size: int, device: str) -> torch.Tensor:
    kernels = []
    for zid in [0, 1, 2]:
        kernels.append(build_kernel_np(zone_specs[zid], kernel_size))
    kernels = np.stack(kernels, axis=0)
    return torch.tensor(kernels, dtype=torch.float32, device=device)  # [3,K,K]


def apply_pdk_soft_torch(img: torch.Tensor, zone_probs: torch.Tensor, kernel_bank: torch.Tensor) -> torch.Tensor:
    # img: [B,C,H,W], zone_probs: [B,3,H,W], kernel_bank: [3,K,K]
    filtered = []
    for i in range(3):
        out_i = _depthwise_conv(img, kernel_bank[i])
        filtered.append(out_i)
    filtered = torch.stack(filtered, dim=1)  # [B,3,C,H,W]
    probs = zone_probs.unsqueeze(2)          # [B,3,1,H,W]
    out = (filtered * probs).sum(dim=1)
    return out.clamp(0.0, 1.0)


def apply_pdk_hard_torch(img: torch.Tensor, zone_map: torch.Tensor, kernel_bank: torch.Tensor) -> torch.Tensor:
    filtered = []
    for i in range(3):
        out_i = _depthwise_conv(img, kernel_bank[i])
        filtered.append(out_i)
    filtered = torch.stack(filtered, dim=1)  # [B,3,C,H,W]
    onehot = F.one_hot(zone_map.long(), num_classes=3).permute(0, 3, 1, 2).float().unsqueeze(2)
    out = (filtered * onehot).sum(dim=1)
    return out.clamp(0.0, 1.0)
