import os
import numpy as np
import matplotlib.pyplot as plt


def save_image(path: str, img: np.ndarray, cmap: str = "gray", vmin: float = 0.0, vmax: float = 1.0):
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


def save_fovea_panel(
    save_path: str,
    clean: np.ndarray,
    observed: np.ndarray,
    global_out: np.ndarray,
    pdk_out: np.ndarray,
    fovea_map: np.ndarray,
    zone_map: np.ndarray,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    err_global = np.abs(global_out - clean)
    err_pdk = np.abs(pdk_out - clean)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Clean")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(observed, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Observed")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(fovea_map, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("Fovea Score")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(zone_map, cmap="viridis", vmin=0, vmax=2)
    axes[0, 3].set_title("Foveated 3-zone")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(global_out, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Best Global")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pdk_out, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("NN-Foveated PDK")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(err_global, cmap="inferno")
    axes[1, 2].set_title("Error: Global")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(err_pdk, cmap="inferno")
    axes[1, 3].set_title("Error: PDK")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_barplot(path: str, metric_dict: dict, ylabel: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = list(metric_dict.keys())
    values = [metric_dict[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()