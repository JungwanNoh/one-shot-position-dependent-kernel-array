import os
import csv
import numpy as np
from PIL import Image

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


def load_gaze_dict(csv_path: str):
    gaze_dict = {}
    if not os.path.exists(csv_path):
        return gaze_dict

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            x = float(row["x_norm"])
            y = float(row["y_norm"])
            gaze_dict[name] = (x, y)
    return gaze_dict


def build_samples(root: str, max_images: int | None, base_seed: int):
    paths = list_image_files(root, max_images=max_images)
    gaze_dict = load_gaze_dict(CFG.GAZE_CSV_PATH)

    samples = []
    for idx, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        clean = load_grayscale(path, CFG.IMAGE_SIZE)

        rng = np.random.default_rng(base_seed + idx)
        observed, v_map, sigma_map = generate_observed_image(clean, rng)

        gaze_xy = gaze_dict.get(name, CFG.DEFAULT_GAZE)

        samples.append({
            "name": name,
            "clean": clean,
            "observed": observed,
            "v_map": v_map,
            "sigma_map": sigma_map,
            "gaze_xy": gaze_xy,
        })
    return samples