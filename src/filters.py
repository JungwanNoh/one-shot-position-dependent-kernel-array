import numpy as np
from scipy.ndimage import convolve

from config import CFG
from utils import clip01


def identity_kernel(kernel_size: int) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    k[c, c] = 1.0
    return k


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2 + 1e-12))
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def binomial_kernel(kernel_size: int) -> np.ndarray:
    order = kernel_size - 1
    vec = np.array([1.0], dtype=np.float32)
    for _ in range(order):
        vec = np.convolve(vec, np.array([1.0, 1.0], dtype=np.float32)).astype(np.float32)
    kernel = np.outer(vec, vec)
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def unsharp_kernel(kernel_size: int, sigma: float, amount: float) -> np.ndarray:
    delta = identity_kernel(kernel_size)
    blur = gaussian_kernel(kernel_size, sigma)
    kernel = delta + amount * (delta - blur)
    return kernel.astype(np.float32)


def build_kernel(spec: dict, kernel_size: int | None = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    family = spec["family"]
    if family == "gaussian":
        return gaussian_kernel(kernel_size, sigma=float(spec["sigma"]))
    if family == "binomial":
        return binomial_kernel(kernel_size)
    if family == "unsharp":
        return unsharp_kernel(kernel_size, sigma=float(spec["sigma"]), amount=float(spec["amount"]))
    raise ValueError(f"Unknown kernel family: {family}")


def apply_global_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(CFG.KERNEL_SIZE, sigma)
    out = convolve(img, kernel, mode="reflect")
    return clip01(out)


def apply_zonewise_pdk(img: np.ndarray, zone_map: np.ndarray, zone_specs: dict) -> np.ndarray:
    filtered_bank = {}
    for zone_id in [0, 1, 2]:
        kernel = build_kernel(zone_specs[zone_id], CFG.KERNEL_SIZE)
        filtered_bank[zone_id] = convolve(img, kernel, mode="reflect").astype(np.float32)

    out = np.zeros_like(img, dtype=np.float32)
    for zone_id in [0, 1, 2]:
        mask = zone_map == zone_id
        out[mask] = filtered_bank[zone_id][mask]

    return clip01(out)