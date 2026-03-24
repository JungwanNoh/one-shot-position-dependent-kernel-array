import numpy as np
from config import CFG
from utils import clip01_np


def generate_coordinate_grid(h: int, w: int):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def generate_variability_map(h: int, w: int, mode: str | None = None) -> np.ndarray:
    if mode is None:
        mode = CFG.MAP_MODE

    yy, xx = generate_coordinate_grid(h, w)

    if mode == "radial":
        cx, cy = CFG.RADIAL_CENTER
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rr = rr / (rr.max() + 1e-8)
        falloff = rr ** CFG.RADIAL_FALLOFF
        v_map = CFG.V_MAX - (CFG.V_MAX - CFG.V_MIN) * falloff

    elif mode == "blocky":
        v_map = np.ones((h, w), dtype=np.float32) * 0.95
        h1, h2 = h // 3, 2 * h // 3
        w1, w2 = w // 3, 2 * w // 3

        v_map[:h1, :w1] = 0.55
        v_map[:h1, w2:] = 0.65
        v_map[h2:, :w1] = 0.60
        v_map[h2:, w2:] = 0.50
        v_map[h1:h2, w1:w2] = 0.95
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return clip01_np(v_map)


def generate_noise_sigma_map(v_map: np.ndarray) -> np.ndarray:
    sigma_map = CFG.NOISE_SIGMA_MIN + CFG.NOISE_SIGMA_ALPHA * (1.0 - v_map)
    return sigma_map.astype(np.float32)


def generate_observed_image(clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = clean.shape
    v_map = generate_variability_map(h, w, mode=CFG.MAP_MODE)
    sigma_map = generate_noise_sigma_map(v_map)
    noise = rng.normal(loc=0.0, scale=sigma_map, size=clean.shape).astype(np.float32)

    observed = v_map * clean + noise
    observed = clip01_np(observed)

    return observed, v_map, sigma_map