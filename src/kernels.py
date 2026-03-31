import numpy as np
from scipy.ndimage import convolve
from config import KERNEL_SIZE, TASK_ZONE_KERNEL_SPECS, GLOBAL_KERNEL_SPECS, ZONE_LABELS


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


def box_kernel_np(kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel.sum() + 1e-12
    return kernel


def unsharp_kernel_np(kernel_size: int, sigma: float, amount: float) -> np.ndarray:
    delta = identity_kernel_np(kernel_size)
    blur = gaussian_kernel_np(kernel_size, sigma)
    kernel = delta + amount * (delta - blur)
    return kernel.astype(np.float32)


def highboost_kernel_np(kernel_size: int, alpha: float) -> np.ndarray:
    delta = identity_kernel_np(kernel_size)
    blur = binomial_kernel_np(kernel_size)
    kernel = delta + alpha * (delta - blur)
    return kernel.astype(np.float32)


def laplacian_sharpen_kernel_np(kernel_size: int, beta: float) -> np.ndarray:
    if kernel_size != 3:
        raise ValueError('laplacian_sharpen currently supports only k=3')
    lap = np.array([[0.0, -1.0, 0.0],
                    [-1.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0]], dtype=np.float32)
    kernel = identity_kernel_np(kernel_size) + beta * lap
    return kernel.astype(np.float32)


def build_kernel_np(spec: dict, kernel_size: int | None = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = KERNEL_SIZE
    family = spec['family']
    if family == 'gaussian':
        return gaussian_kernel_np(kernel_size, float(spec['sigma']))
    if family == 'binomial':
        return binomial_kernel_np(kernel_size)
    if family == 'box':
        return box_kernel_np(kernel_size)
    if family == 'unsharp':
        return unsharp_kernel_np(kernel_size, float(spec['sigma']), float(spec['amount']))
    if family == 'highboost':
        return highboost_kernel_np(kernel_size, float(spec['alpha']))
    if family == 'laplacian_sharpen':
        return laplacian_sharpen_kernel_np(kernel_size, float(spec['beta']))
    if family == 'identity':
        return identity_kernel_np(kernel_size)
    raise ValueError(f'Unknown kernel family: {family}')


def apply_kernel_np(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = convolve(img, kernel, mode='reflect')
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def get_global_kernel_spec(task_name: str) -> dict:
    return GLOBAL_KERNEL_SPECS[task_name]


def get_global_kernel_matrix(task_name: str) -> np.ndarray:
    return build_kernel_np(get_global_kernel_spec(task_name), KERNEL_SIZE)


def get_zone_kernel_specs(task_name: str) -> dict:
    return TASK_ZONE_KERNEL_SPECS[task_name]


def get_zone_kernel_matrices(task_name: str) -> dict[int, np.ndarray]:
    zone_specs = get_zone_kernel_specs(task_name)
    return {zone_id: build_kernel_np(zone_specs[zone_id], KERNEL_SIZE) for zone_id in [0, 1, 2]}


def apply_global_kernel_np(img: np.ndarray, task_name: str) -> np.ndarray:
    kernel = get_global_kernel_matrix(task_name)
    return apply_kernel_np(img, kernel)


def apply_zonewise_pdk_np(img: np.ndarray, zone_map: np.ndarray, task_name: str) -> np.ndarray:
    zone_specs = get_zone_kernel_specs(task_name)
    bank = {}
    for zone_id in [0, 1, 2]:
        kernel = build_kernel_np(zone_specs[zone_id], KERNEL_SIZE)
        bank[zone_id] = apply_kernel_np(img, kernel)
    out = np.zeros_like(img, dtype=np.float32)
    for zone_id in [0, 1, 2]:
        mask = zone_map == zone_id
        out[mask] = bank[zone_id][mask]
    return out


def format_kernel_spec(spec: dict) -> str:
    family = spec.get('family', 'unknown')
    if family == 'gaussian':
        return f"gaussian(sigma={float(spec['sigma']):.2f}, k={KERNEL_SIZE})"
    if family == 'binomial':
        return f"binomial(k={KERNEL_SIZE})"
    if family == 'box':
        return f"box(k={KERNEL_SIZE})"
    if family == 'unsharp':
        return f"unsharp(sigma={float(spec['sigma']):.2f}, amount={float(spec['amount']):.2f}, k={KERNEL_SIZE})"
    if family == 'highboost':
        return f"highboost(alpha={float(spec['alpha']):.2f}, base=binomial, k={KERNEL_SIZE})"
    if family == 'laplacian_sharpen':
        return f"laplacian(beta={float(spec['beta']):.2f}, k={KERNEL_SIZE})"
    if family == 'identity':
        return f"identity(k={KERNEL_SIZE})"
    return str(spec)


def format_zone_kernel_specs(task_name: str) -> str:
    zone_specs = get_zone_kernel_specs(task_name)
    parts = []
    for zone_id in [0, 1, 2]:
        parts.append(f"{ZONE_LABELS[zone_id]}={format_kernel_spec(zone_specs[zone_id])}")
    return ' | '.join(parts)
