import json
import numpy as np
from scipy.ndimage import convolve, uniform_filter, median_filter

from config import CFG
from utils import clip01
from metrics import gradient_magnitude


def identity_kernel(kernel_size: int) -> np.ndarray:
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    c = kernel_size // 2
    k[c, c] = 1.0
    return k


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2 + 1e-12))
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def binomial_kernel(kernel_size: int) -> np.ndarray:
    """
    Fixed-support Pascal/binomial low-pass kernel.
    kernel_size=5 -> order 4
    kernel_size=7 -> order 6
    """
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


def dog_highboost_kernel(kernel_size: int, sigma_small: float, sigma_large: float, amount: float) -> np.ndarray:
    """
    Difference-of-Gaussians high-boost style kernel.
    Sum remains ~1, more edge emphasis than plain gaussian.
    """
    delta = identity_kernel(kernel_size)
    g_small = gaussian_kernel(kernel_size, sigma_small)
    g_large = gaussian_kernel(kernel_size, sigma_large)
    kernel = delta + amount * (g_small - g_large)
    return kernel.astype(np.float32)


def build_kernel_from_spec(spec: dict, kernel_size: int = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    family = spec["family"]

    if family == "identity":
        return identity_kernel(kernel_size)

    if family == "gaussian":
        return gaussian_kernel(kernel_size, sigma=float(spec["sigma"]))

    if family == "binomial":
        return binomial_kernel(kernel_size)

    if family == "unsharp":
        return unsharp_kernel(
            kernel_size=kernel_size,
            sigma=float(spec["sigma"]),
            amount=float(spec["amount"]),
        )

    if family == "dog_highboost":
        return dog_highboost_kernel(
            kernel_size=kernel_size,
            sigma_small=float(spec["sigma_small"]),
            sigma_large=float(spec["sigma_large"]),
            amount=float(spec["amount"]),
        )

    raise ValueError(f"Unknown kernel family: {family}")


def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = convolve(img, kernel, mode="reflect")
    return clip01(out)


def apply_global_filter(img: np.ndarray, sigma: float, kernel_size: int = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE
    kernel = gaussian_kernel(kernel_size, sigma)
    return apply_kernel(img, kernel)


def spec_to_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def local_variance_map(img: np.ndarray, win: int = 5) -> np.ndarray:
    mean = uniform_filter(img, size=win, mode="reflect")
    mean_sq = uniform_filter(img * img, size=win, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return var.astype(np.float32)


def generate_content_map(observed: np.ndarray) -> np.ndarray:
    """
    content 0: flat
    content 1: texture
    content 2: edge

    Observed만 보고 계산되도록 설계.
    noise 민감도를 줄이기 위해 먼저 약하게 pre-smoothing 한 뒤 gradient/variance 추출.
    """
    pre = apply_global_filter(observed, sigma=CFG.CONTENT_PRE_SMOOTH_SIGMA, kernel_size=CFG.KERNEL_SIZE)

    grad = gradient_magnitude(pre)
    lvar = local_variance_map(pre, win=CFG.CONTENT_LOCAL_VAR_WIN)

    th_edge = np.percentile(grad, CFG.CONTENT_EDGE_PERCENTILE)
    th_tex_g = np.percentile(grad, CFG.CONTENT_TEXTURE_PERCENTILE)
    th_tex_v = np.percentile(lvar, CFG.CONTENT_TEXTURE_PERCENTILE)

    content_map = np.zeros_like(observed, dtype=np.int32)

    texture_mask = (grad >= th_tex_g) | (lvar >= th_tex_v)
    edge_mask = grad >= th_edge

    content_map[texture_mask] = 1
    content_map[edge_mask] = 2

    # state flicker 방지용 median stabilization
    content_map = median_filter(content_map, size=CFG.CONTENT_MEDIAN_SIZE, mode="nearest")

    return content_map.astype(np.int32)


def apply_content_aware_pdk(
    img: np.ndarray,
    region_map: np.ndarray,
    content_map: np.ndarray,
    lut: dict,
    kernel_size: int = None,
) -> np.ndarray:
    """
    Practical discrete LUT:
    (region, content) -> kernel spec
    """
    if kernel_size is None:
        kernel_size = CFG.KERNEL_SIZE

    out = np.zeros_like(img, dtype=np.float32)

    filtered_bank = {}
    for _, spec in lut.items():
        key = spec_to_key(spec)
        if key not in filtered_bank:
            kernel = build_kernel_from_spec(spec, kernel_size=kernel_size)
            filtered_bank[key] = apply_kernel(img, kernel)

    for region_id in [0, 1, 2]:
        for content_id in [0, 1, 2]:
            spec = lut[(region_id, content_id)]
            key = spec_to_key(spec)
            mask = (region_map == region_id) & (content_map == content_id)
            out[mask] = filtered_bank[key][mask]

    return clip01(out)