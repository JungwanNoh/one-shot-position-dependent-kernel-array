import numpy as np
from scipy.ndimage import sobel, gaussian_filter, uniform_filter
from utils import normalize01
from config import ZONE


def gradient_mag_np(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode='reflect')
    gy = sobel(img, axis=0, mode='reflect')
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def local_variance_np(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    mean = gaussian_filter(img, sigma=sigma, mode='reflect')
    mean_sq = gaussian_filter(img * img, sigma=sigma, mode='reflect')
    return np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32)


def local_mean_np(img: np.ndarray, size: int = 7) -> np.ndarray:
    return uniform_filter(img.astype(np.float32), size=size, mode='reflect')


def local_mad_np(img: np.ndarray, size: int = 7) -> np.ndarray:
    mean = local_mean_np(img, size=size)
    mad = uniform_filter(np.abs(img - mean).astype(np.float32), size=size, mode='reflect')
    return mad.astype(np.float32)


def structure_tensor_coherence_np(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    gx = sobel(img, axis=1, mode='reflect').astype(np.float32)
    gy = sobel(img, axis=0, mode='reflect').astype(np.float32)
    jxx = gaussian_filter(gx * gx, sigma=sigma, mode='reflect')
    jyy = gaussian_filter(gy * gy, sigma=sigma, mode='reflect')
    jxy = gaussian_filter(gx * gy, sigma=sigma, mode='reflect')
    tmp = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2))
    l1 = 0.5 * (jxx + jyy + tmp)
    l2 = 0.5 * (jxx + jyy - tmp)
    coherence = (l1 - l2) / (l1 + l2 + 1e-8)
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def random_field_map(h: int, w: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    field = gaussian_filter(field, sigma=sigma, mode='reflect')
    return normalize01(field)


def missed_edge_map(raw: np.ndarray, global_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_raw = normalize01(gradient_mag_np(raw))
    g_global = normalize01(gradient_mag_np(global_out))
    missed = np.maximum(g_raw - g_global, 0.0).astype(np.float32)
    return g_raw, g_global, normalize01(missed)


def synthetic_scores_np(raw: np.ndarray, global_out: np.ndarray, missed_w: float, coherence_w: float):
    _, g_global, missed = missed_edge_map(raw, global_out)
    coherence = normalize01(structure_tensor_coherence_np(raw, sigma=1.1))
    recover = normalize01(missed_w * missed + coherence_w * coherence)
    preserve = normalize01(g_global * coherence)
    suppress = normalize01(np.clip(1.0 - np.maximum(recover, preserve), 0.0, 1.0))
    recover = normalize01(gaussian_filter(recover, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    preserve = normalize01(gaussian_filter(preserve, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    suppress = normalize01(gaussian_filter(suppress, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    return preserve, recover, suppress


def lowlight_scores_np(raw: np.ndarray, global_out: np.ndarray, missed_w: float, coherence_w: float, brightness_w: float, noise_w: float):
    _, g_global, missed = missed_edge_map(raw, global_out)
    coherence = normalize01(structure_tensor_coherence_np(raw, sigma=1.2))
    brightness = normalize01(local_mean_np(raw, size=9))
    noise = normalize01(local_mad_np(raw, size=7))
    recover = normalize01(missed_w * missed + coherence_w * coherence + brightness_w * brightness - noise_w * noise)
    preserve = normalize01(g_global * coherence)
    suppress = normalize01(0.6 * noise + 0.4 * (1.0 - brightness))
    recover = normalize01(gaussian_filter(recover, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    preserve = normalize01(gaussian_filter(preserve, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    suppress = normalize01(gaussian_filter(suppress, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    return preserve, recover, suppress


def natural_scores_np(raw: np.ndarray, global_out: np.ndarray, missed_w: float, coherence_w: float, texture_penalty_w: float):
    _, g_global, missed = missed_edge_map(raw, global_out)
    coherence = normalize01(structure_tensor_coherence_np(raw, sigma=1.2))
    texture = normalize01(local_variance_np(raw, sigma=1.8))
    recover = normalize01(missed_w * missed + coherence_w * coherence - texture_penalty_w * texture)
    preserve = normalize01(g_global * coherence)
    suppress = normalize01((1.0 - coherence) + 0.2 * texture)
    recover = normalize01(gaussian_filter(recover, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    preserve = normalize01(gaussian_filter(preserve, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    suppress = normalize01(gaussian_filter(suppress, sigma=ZONE['score_blur_sigma'], mode='reflect'))
    return preserve, recover, suppress


def assign_zones_from_scores(preserve: np.ndarray, recover: np.ndarray, suppress: np.ndarray) -> np.ndarray:
    stacked = np.stack([preserve, recover, suppress], axis=0)
    zone = np.argmax(stacked, axis=0).astype(np.int32)
    return zone


def edge_strength_synthetic(img: np.ndarray) -> np.ndarray:
    return normalize01(gradient_mag_np(img))


def edge_strength_lowlight(img: np.ndarray) -> np.ndarray:
    grad = normalize01(gradient_mag_np(img))
    brightness = normalize01(local_mean_np(img, size=9))
    weighted = grad * (0.35 + 0.65 * brightness)
    return normalize01(weighted)


def edge_strength_natural(img: np.ndarray) -> np.ndarray:
    grad = normalize01(gradient_mag_np(img))
    coherence = normalize01(structure_tensor_coherence_np(img, sigma=1.2))
    weighted = grad * (0.25 + 0.75 * coherence)
    return normalize01(weighted)


def edge_response_synthetic(processed: np.ndarray) -> np.ndarray:
    return edge_strength_synthetic(processed)


def edge_response_lowlight(processed: np.ndarray) -> np.ndarray:
    return edge_strength_lowlight(processed)


def edge_response_natural(processed: np.ndarray) -> np.ndarray:
    return edge_strength_natural(processed)


def target_edge_map(img: np.ndarray) -> np.ndarray:
    return normalize01(gradient_mag_np(img))
