import os
import random
import numpy as np
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def list_image_files(root: str, exts, max_images=None):
    if not os.path.exists(root):
        return []
    files = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            files.append(path)
    files.sort()
    if max_images is not None:
        files = files[:max_images]
    return files


def load_grayscale(path: str, image_size):
    img = Image.open(path).convert("L")
    img = img.resize(image_size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.astype(np.float32)