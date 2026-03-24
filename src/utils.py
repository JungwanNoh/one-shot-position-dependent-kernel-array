import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_to_unit(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)