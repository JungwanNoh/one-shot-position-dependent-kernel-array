import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from config import CFG
from variability import generate_observed_image


def list_image_files(root: str, max_images: int | None = None):
    files = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(CFG.FILE_EXTENSIONS):
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


class BSDSAttentionDataset(Dataset):
    def __init__(self, paths, image_size, base_seed: int):
        self.paths = paths
        self.image_size = image_size
        self.base_seed = base_seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        stem = os.path.splitext(os.path.basename(path))[0]

        clean = load_grayscale(path, self.image_size)

        rng = np.random.default_rng(self.base_seed + idx)
        observed, v_map, sigma_map = generate_observed_image(clean, rng)

        clean_t = torch.from_numpy(clean).unsqueeze(0)      # [1,H,W]
        observed_t = torch.from_numpy(observed).unsqueeze(0)
        v_map_t = torch.from_numpy(v_map).unsqueeze(0)
        sigma_map_t = torch.from_numpy(sigma_map).unsqueeze(0)

        return {
            "name": stem,
            "clean": clean_t,
            "observed": observed_t,
            "v_map": v_map_t,
            "noise_sigma_map": sigma_map_t,
        }