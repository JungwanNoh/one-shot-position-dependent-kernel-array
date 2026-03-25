import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from utils import normalize01
from config import CFG


def gradient_mag(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def quantize_score_to_zone(score: np.ndarray, high_percentile: float | None = None, mid_percentile: float | None = None):
    if high_percentile is None:
        high_percentile = CFG.SCORE_HIGH_PERCENTILE
    if mid_percentile is None:
        mid_percentile = CFG.SCORE_MID_PERCENTILE

    hi = np.percentile(score, high_percentile)
    md = np.percentile(score, mid_percentile)

    zone = np.full_like(score, 2, dtype=np.int32)  # around
    zone[score >= md] = 1                           # intermediate
    zone[score >= hi] = 0                           # attention
    return zone


def synthetic_variability_map(h: int, w: int, mode="radial", vmin=0.45, vmax=1.0, center=(0.5, 0.5), falloff=1.6):
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
        v = vmax - (vmax - vmin) * fall
    else:
        v = np.ones((h, w), dtype=np.float32) * 0.95
        h1, h2 = h // 3, 2 * h // 3
        w1, w2 = w // 3, 2 * w // 3
        v[:h1, :w1] = 0.55
        v[:h1, w2:] = 0.65
        v[h2:, :w1] = 0.60
        v[h2:, w2:] = 0.50
        v[h1:h2, w1:w2] = 0.95
    return v.astype(np.float32)


def global_first_score(observed: np.ndarray, global_out: np.ndarray, residual_w=0.4, gradient_w=0.3, extra_map=None, extra_w=0.3):
    residual = normalize01(np.abs(observed - global_out))
    grad = normalize01(gradient_mag(global_out))

    score = residual_w * residual + gradient_w * grad
    if extra_map is not None:
        score = score + extra_w * normalize01(extra_map)
    return normalize01(score)


def lowlight_score(observed: np.ndarray, global_out: np.ndarray, illum_sigma: float, weights):
    residual_w, gradient_w, lowlight_w = weights
    illum = gaussian_filter(observed, sigma=illum_sigma)
    lowlightness = 1.0 - normalize01(illum)
    return global_first_score(
        observed,
        global_out,
        residual_w=residual_w,
        gradient_w=gradient_w,
        extra_map=lowlightness,
        extra_w=lowlight_w,
    )


def natural_score(observed: np.ndarray, global_out: np.ndarray, weights):
    residual_w, edge_w = weights
    residual = normalize01(np.abs(observed - global_out))
    edge = normalize01(gradient_mag(observed))
    score = residual_w * residual + edge_w * edge
    return normalize01(score)