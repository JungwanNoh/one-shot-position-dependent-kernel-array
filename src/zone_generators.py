import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from utils import normalize01


def generate_variability_map(h, w, mode="radial", center=(0.5, 0.5), falloff=1.6, v_max=1.0, v_min=0.45):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    if mode == "radial":
        cx, cy = center
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rr = rr / (rr.max() + 1e-8)
        fall = rr ** falloff
        v = v_max - (v_max - v_min) * fall
    elif mode == "blocky":
        v = np.ones((h, w), dtype=np.float32) * 0.95
        h1, h2 = h // 3, 2 * h // 3
        w1, w2 = w // 3, 2 * w // 3
        v[:h1, :w1] = 0.55
        v[:h1, w2:] = 0.65
        v[h2:, :w1] = 0.60
        v[h2:, w2:] = 0.50
        v[h1:h2, w1:w2] = 0.95
    else:
        raise ValueError(mode)

    return np.clip(v, 0.0, 1.0).astype(np.float32)


def observed_from_clean(clean, v_map, noise_sigma_min=0.01, noise_sigma_alpha=0.08, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sigma_map = noise_sigma_min + noise_sigma_alpha * (1.0 - v_map)
    noise = rng.normal(0.0, sigma_map, size=clean.shape).astype(np.float32)
    observed = np.clip(v_map * clean + noise, 0.0, 1.0).astype(np.float32)
    return observed, sigma_map.astype(np.float32)


def zone_from_reliability_map(v_map, th_high=0.78, th_mid=0.58):
    z = np.zeros_like(v_map, dtype=np.int32)
    z[v_map >= th_high] = 0
    mid = (v_map < th_high) & (v_map >= th_mid)
    z[mid] = 1
    z[v_map < th_mid] = 2
    return z


def illumination_map(gray: np.ndarray) -> np.ndarray:
    illum = gaussian_filter(gray, sigma=3.0)
    return normalize01(illum)


def zone_from_illumination(gray: np.ndarray, high=0.62, mid=0.38) -> tuple[np.ndarray, np.ndarray]:
    illum = illumination_map(gray)
    z = np.zeros_like(illum, dtype=np.int32)
    z[illum >= high] = 0
    m = (illum < high) & (illum >= mid)
    z[m] = 1
    z[illum < mid] = 2
    return illum, z


def edge_energy_center(gray: np.ndarray) -> tuple[float, float, np.ndarray]:
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    mag = gaussian_filter(mag, sigma=4.0)
    mag = normalize01(mag)

    h, w = gray.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    total = mag.sum() + 1e-8
    cx = float((mag * xx).sum() / total)
    cy = float((mag * yy).sum() / total)
    return cx, cy, mag


def foveated_zone_from_center(h: int, w: int, cx: float, cy: float, r1: float, r2: float):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    z = np.full((h, w), 2, dtype=np.int32)
    z[dist <= r2] = 1
    z[dist <= r1] = 0
    return dist, z
