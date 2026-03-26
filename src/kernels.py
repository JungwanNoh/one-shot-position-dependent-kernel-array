import numpy as np
from scipy.ndimage import convolve
from config import KERNEL_SIZE, ZONE_KERNEL_SPECS

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

def build_kernel_np(spec: dict, kernel_size: int | None = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = KERNEL_SIZE
    family = spec['family']
    if family == 'gaussian':
        return gaussian_kernel_np(kernel_size, float(spec['sigma']))
    if family == 'binomial':
        return binomial_kernel_np(kernel_size)
    if family == 'unsharp':
        return unsharp_kernel_np(kernel_size, float(spec['sigma']), float(spec['amount']))
    raise ValueError(f'Unknown kernel family: {family}')

def apply_kernel_np(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = convolve(img, kernel, mode='reflect')
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def apply_global_gaussian_np(img: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel_np(KERNEL_SIZE, sigma)
    return apply_kernel_np(img, k)

def apply_zonewise_pdk_np(img: np.ndarray, zone_map: np.ndarray, zone_specs: dict | None = None) -> np.ndarray:
    if zone_specs is None:
        zone_specs = ZONE_KERNEL_SPECS
    bank = {}
    for zone_id in [0, 1, 2]:
        kernel = build_kernel_np(zone_specs[zone_id], KERNEL_SIZE)
        bank[zone_id] = apply_kernel_np(img, kernel)
    out = np.zeros_like(img, dtype=np.float32)
    for zone_id in [0, 1, 2]:
        mask = zone_map == zone_id
        out[mask] = bank[zone_id][mask]
    return out

def get_torch_device(preferred='cuda'):
    import torch
    if preferred == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def apply_global_gaussian_torch(x, sigma: float):
    import torch
    import torch.nn.functional as F
    c = x.shape[1]
    k = gaussian_kernel_np(KERNEL_SIZE, sigma)
    k = torch.tensor(k, dtype=torch.float32, device=x.device)[None, None, :, :].repeat(c, 1, 1, 1)
    pad = KERNEL_SIZE // 2
    out = F.conv2d(x, k, padding=pad, groups=c)
    return out.clamp(0.0, 1.0)

def kernel_bank_torch(zone_specs: dict, in_channels: int, device: str):
    import torch
    bank = []
    for zone_id in [0, 1, 2]:
        k = build_kernel_np(zone_specs[zone_id], KERNEL_SIZE)
        k = torch.tensor(k, dtype=torch.float32, device=device)[None, None, :, :]
        k = k.repeat(in_channels, 1, 1, 1)
        bank.append(k)
    return bank

def apply_zonewise_pdk_hard_torch(x, zone_map, bank):
    import torch
    import torch.nn.functional as F
    pad = KERNEL_SIZE // 2
    filtered = []
    for k in bank:
        filtered.append(F.conv2d(x, k, padding=pad, groups=x.shape[1]))
    filtered = torch.stack(filtered, dim=1)  # [B,3,C,H,W]
    onehot = F.one_hot(zone_map.long(), num_classes=3).permute(0, 3, 1, 2).float()
    onehot = onehot.unsqueeze(2)
    out = (filtered * onehot).sum(dim=1)
    return out.clamp(0.0, 1.0)
