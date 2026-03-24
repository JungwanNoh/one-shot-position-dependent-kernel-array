import os
from typing import List, Tuple
import numpy as np
from PIL import Image

from config import CFG
from utils import clip01


def list_image_files(root: str) -> List[str]:
    files = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(CFG.FILE_EXTENSIONS):
            files.append(path)
    files.sort()
    return files[:CFG.MAX_IMAGES]


def load_grayscale_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(image_size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return clip01(arr)


def load_dataset() -> List[Tuple[str, np.ndarray]]:
    image_files = list_image_files(CFG.DATA_ROOT)
    data = []
    for path in image_files:
        img = load_grayscale_image(path, CFG.IMAGE_SIZE)
        stem = os.path.splitext(os.path.basename(path))[0]
        data.append((stem, img))
    return data