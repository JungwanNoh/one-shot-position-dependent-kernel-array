import numpy as np
import torch
import torch.nn.functional as F

from config import CFG


def identity_kernel(kernel_size: int) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    k[c, c] = 1.0
    return k


def gaussian_kernel_np(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2 + 1e-12))
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
    delta = identity_kernel(kernel_size)
    blur = gaussian_kernel_np(kernel_size, sigma)
    kernel = delta + amount * (delta - blur)
    return kernel.astype(np.float32)


def build_kernel_np(spec: dict, kernel_size: int | None = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    family = spec["family"]
    if family == "gaussian":
        return gaussian_kernel_np(kernel_size, sigma=float(spec["sigma"]))
    if family == "binomial":
        return binomial_kernel_np(kernel_size)
    if family == "unsharp":
        return unsharp_kernel_np(kernel_size, sigma=float(spec["sigma"]), amount=float(spec["amount"]))
    raise ValueError(f"Unknown kernel family: {family}")


def get_zone_kernel_bank(device: str) -> torch.Tensor:
    kernels = []
    for zone_id in [0, 1, 2]:
        spec = CFG.ZONE_KERNEL_SPECS[zone_id]
        kernels.append(build_kernel_np(spec, CFG.KERNEL_SIZE))
    kernels = np.stack(kernels, axis=0)[:, None, :, :]
    return torch.tensor(kernels, dtype=torch.float32, device=device)


def apply_kernel_bank_soft(img: torch.Tensor, zone_probs: torch.Tensor, kernel_bank: torch.Tensor) -> torch.Tensor:
    k = kernel_bank.shape[-1]
    pad = k // 2

    filtered_list = []
    for i in range(kernel_bank.shape[0]):
        out_i = F.conv2d(img, kernel_bank[i:i+1], padding=pad)
        filtered_list.append(out_i)
    filtered = torch.cat(filtered_list, dim=1)  # [B,3,H,W]

    pred = (filtered * zone_probs).sum(dim=1, keepdim=True)
    return pred.clamp(0.0, 1.0)


def apply_kernel_bank_hard(img: torch.Tensor, zone_map: torch.Tensor, kernel_bank: torch.Tensor) -> torch.Tensor:
    k = kernel_bank.shape[-1]
    pad = k // 2

    filtered_list = []
    for i in range(kernel_bank.shape[0]):
        out_i = F.conv2d(img, kernel_bank[i:i+1], padding=pad)
        filtered_list.append(out_i)
    filtered = torch.cat(filtered_list, dim=1)

    onehot = F.one_hot(zone_map.long(), num_classes=3).permute(0, 3, 1, 2).float()
    pred = (filtered * onehot).sum(dim=1, keepdim=True)
    return pred.clamp(0.0, 1.0)


def apply_global_gaussian(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_np = gaussian_kernel_np(CFG.KERNEL_SIZE, sigma)
    kernel = torch.tensor(kernel_np, dtype=torch.float32, device=img.device)[None, None, :, :]
    pad = CFG.KERNEL_SIZE // 2
    out = F.conv2d(img, kernel, padding=pad)
    return out.clamp(0.0, 1.0)