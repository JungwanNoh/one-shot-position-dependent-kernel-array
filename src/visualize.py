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


def save_heatmap(path: str, img: np.ndarray, cmap='magma', vmin=None, vmax=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def _show_panel_item(ax, title: str, img: np.ndarray, kind: str):
    if kind == 'gray':
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
    elif kind == 'zone':
        ax.imshow(img, cmap='viridis', vmin=0, vmax=2)
    elif kind == 'improvement':
        vmax = float(np.max(np.abs(img)))
        if vmax < 1e-8:
            vmax = 1e-8
        ax.imshow(img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    elif kind == 'delta':
        vmax = float(np.max(np.abs(img)))
        if vmax < 1e-8:
            vmax = 1e-8
        ax.imshow(img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    else:
        ax.imshow(img, cmap='magma')
    ax.set_title(title)
    ax.axis('off')


def save_panel(path: str, raw: np.ndarray, global_out: np.ndarray, pdk_out: np.ndarray, aux_map: np.ndarray, zone: np.ndarray, gt: np.ndarray | None = None, aux_title: str = 'Improvement', aux_kind: str = 'improvement'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if gt is None:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        items = [
            ('Raw', raw, 'gray'),
            ('Global', global_out, 'gray'),
            ('PDK', pdk_out, 'gray'),
            (aux_title, aux_map, aux_kind),
            ('Zone', zone, 'zone'),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        items = [
            ('GT', gt, 'gray'),
            ('Raw', raw, 'gray'),
            ('Global', global_out, 'gray'),
            ('PDK', pdk_out, 'gray'),
            (aux_title, aux_map, aux_kind),
            ('Zone', zone, 'zone'),
        ]
    for ax, (title, img, kind) in zip(axes, items):
        _show_panel_item(ax, title, img, kind)
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
