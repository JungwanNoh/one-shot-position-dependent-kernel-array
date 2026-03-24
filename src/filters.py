import numpy as np
from scipy.ndimage import convolve

from config import CFG
from utils import clip01


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2 + 1e-12))
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def apply_global_filter(img: np.ndarray, sigma: float = None, kernel_size: int = None) -> np.ndarray:
    if sigma is None:
        sigma = CFG.GLOBAL_SIGMA
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    kernel = gaussian_kernel(kernel_size, sigma)
    out = convolve(img, kernel, mode="reflect")
    return clip01(out)


def search_best_global_sigma(
    observed: np.ndarray,
    clean: np.ndarray,
    sigma_candidates: list[float],
    kernel_size: int = None,
) -> tuple[float, np.ndarray, float]:
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    best_sigma = None
    best_out = None
    best_mse = float("inf")

    for sigma in sigma_candidates:
        out = apply_global_filter(observed, sigma=sigma, kernel_size=kernel_size)
        mse = float(np.mean((out - clean) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_sigma = sigma
            best_out = out

    return best_sigma, best_out, best_mse


def apply_pdk_filter(
    img: np.ndarray,
    region_map: np.ndarray,
    region_sigmas: dict = None,
    kernel_size: int = None,
) -> np.ndarray:
    if region_sigmas is None:
        region_sigmas = CFG.REGION_SIGMAS
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    out = np.zeros_like(img, dtype=np.float32)

    filtered_bank = {}
    for region_id, sigma in region_sigmas.items():
        kernel = gaussian_kernel(kernel_size, sigma)
        filtered_bank[region_id] = convolve(img, kernel, mode="reflect").astype(np.float32)

    for region_id in region_sigmas.keys():
        mask = (region_map == region_id)
        out[mask] = filtered_bank[region_id][mask]

    return clip01(out)