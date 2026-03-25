import os
import numpy as np
import matplotlib.pyplot as plt


def save_image(path: str, img: np.ndarray, cmap="gray", vmin=0.0, vmax=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_zone_map(path: str, zone_map: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(zone_map, cmap="viridis", vmin=0, vmax=2)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_panel(path: str, raw: np.ndarray, global_out: np.ndarray, pdk_out: np.ndarray, score: np.ndarray, zone: np.ndarray, gt: np.ndarray | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if gt is None:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        items = [
            ("Raw", raw, "gray"),
            ("Global", global_out, "gray"),
            ("PDK", pdk_out, "gray"),
            ("Score", score, "magma"),
            ("Zone", zone, "viridis"),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        err_g = np.abs(global_out - gt)
        err_p = np.abs(pdk_out - gt)
        items = [
            ("GT", gt, "gray"),
            ("Raw", raw, "gray"),
            ("Score", score, "magma"),
            ("Global", global_out, "gray"),
            ("PDK", pdk_out, "gray"),
            ("Zone", zone, "viridis"),
        ]
        axes = axes.ravel()

    for ax, (title, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def save_barplot(path: str, values: dict, ylabel: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = list(values.keys())
    nums = [values[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, nums)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()