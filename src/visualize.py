import os
import numpy as np
import matplotlib.pyplot as plt


def save_image(path: str, img: np.ndarray, cmap='gray', vmin=0.0, vmax=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def save_zone_map(path: str, zone_map: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(zone_map, cmap='viridis', vmin=0, vmax=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def save_panel(path: str, global_out: np.ndarray, pdk_out: np.ndarray, zone: np.ndarray,
               gt: np.ndarray | None = None, input_img: np.ndarray | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if gt is None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        if input_img is None:
            raise ValueError('input_img is required when gt is None')
        delta = pdk_out - global_out
        delta_lim = float(np.max(np.abs(delta)) + 1e-8)
        items = [
            ('Input', input_img, 'gray', 0.0, 1.0),
            ('Global', global_out, 'gray', 0.0, 1.0),
            ('PDK', pdk_out, 'gray', 0.0, 1.0),
            ('PDK - Global', delta, 'coolwarm', -delta_lim, delta_lim),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        gt_minus_global = gt - global_out
        gt_minus_pdk = gt - pdk_out
        resid_lim = float(max(np.max(np.abs(gt_minus_global)), np.max(np.abs(gt_minus_pdk))) + 1e-8)
        items = [
            ('GT', gt, 'gray', 0.0, 1.0),
            ('Global', global_out, 'gray', 0.0, 1.0),
            ('PDK', pdk_out, 'gray', 0.0, 1.0),
            ('Zone', zone, 'viridis', 0, 2),
            ('GT - Global', gt_minus_global, 'coolwarm', -resid_lim, resid_lim),
            ('GT - PDK', gt_minus_pdk, 'coolwarm', -resid_lim, resid_lim),
        ]

    for ax, (title, img, cmap, vmin, vmax) in zip(axes, items):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()


def save_barplot(path: str, values: dict, ylabel: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = list(values.keys())
    nums = [values[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, nums)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
