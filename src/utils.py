import os
import random
from typing import List

import numpy as np
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_image_files(root: str, extensions=(".jpg", ".jpeg", ".png", ".bmp"), max_images=None) -> List[str]:
    files = []
    if not os.path.exists(root):
        return files
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(extensions):
            files.append(path)
    files.sort()
    if max_images is not None:
        files = files[:max_images]
    return files


def load_grayscale(path: str, image_size) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_rgb(path: str, image_size) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = x.min()
    xmax = x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin)
