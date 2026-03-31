import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_image(path: str, img: np.ndarray, cmap='gray', vmin=None, vmax=None):
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


def _draw_kernel_heatmap(ax, kernel: np.ndarray, title: str, annotate: bool = True, fontsize: int = 9):
    k_abs = max(abs(float(kernel.min())), abs(float(kernel.max())), 1e-6)
    ax.imshow(kernel, cmap='coolwarm', vmin=-k_abs, vmax=k_abs)
    if annotate:
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, f'{kernel[i, j]:.2f}', ha='center', va='center', fontsize=fontsize, color='black')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_zone_kernel_triptych(ax, zone_kernel_mats: dict[int, np.ndarray]):
    ax.set_title('Zone → Kernel')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    labels = [
        (0, 'P', 'id'),
        (1, 'R', 'boost'),
        (2, 'S', 'smooth'),
    ]
    lefts = [0.02, 0.35, 0.68]
    width = 0.28
    for left, (zone_id, short_label, full_label) in zip(lefts, labels):
        outer = ax.inset_axes([left, 0.04, width, 0.92])
        outer.set_xticks([])
        outer.set_yticks([])
        outer.set_facecolor('white')
        for spine in outer.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('0.5')
        outer.text(0.5, 0.92, short_label, ha='center', va='center', fontsize=12, transform=outer.transAxes)
        kax = outer.inset_axes([0.10, 0.18, 0.80, 0.68])
        kernel = zone_kernel_mats[zone_id]
        k_abs = max(abs(float(kernel.min())), abs(float(kernel.max())), 1e-6)
        kax.imshow(kernel, cmap='coolwarm', vmin=-k_abs, vmax=k_abs)
        kax.set_xticks([])
        kax.set_yticks([])
        for spine in kax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_edgecolor('0.4')
        outer.text(0.5, 0.07, full_label, ha='center', va='center', fontsize=10, transform=outer.transAxes)


def save_panel_3x3(path: str,
                   global_kernel: np.ndarray,
                   zone_score: np.ndarray,
                   zone_kernel_mats: dict[int, np.ndarray],
                   raw_image: np.ndarray,
                   global_response: np.ndarray,
                   pdk_response: np.ndarray,
                   reference: np.ndarray,
                   global_processed: np.ndarray,
                   pdk_processed: np.ndarray,
                   reference_title: str = 'Reference'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    resp_vmax = max(float(np.max(global_response)), float(np.max(pdk_response)), 1e-6)
    row1 = [
        ('Raw', raw_image, 'gray', 0.0, 1.0),
        ('Global', global_response, 'gray', 0.0, resp_vmax),
        ('PDK', pdk_response, 'gray', 0.0, resp_vmax),
    ]
    row2 = [
        (reference_title, reference, 'gray', 0.0, 1.0),
        ('Global processed', global_processed, 'gray', 0.0, 1.0),
        ('PDK processed', pdk_processed, 'gray', 0.0, 1.0),
    ]

    for ax, (title, img, cmap, vmin, vmax) in zip(axes[0], row1):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')

    for ax, (title, img, cmap, vmin, vmax) in zip(axes[1], row2):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')

    _draw_kernel_heatmap(axes[2, 0], global_kernel, 'Global Kernel', annotate=True, fontsize=9)

    ax = axes[2, 1]
    vmax = max(float(zone_score.max()), 1e-6)
    ax.imshow(zone_score, cmap='magma', vmin=0.0, vmax=vmax)
    ax.set_title('Zone Score (recover)')
    ax.axis('off')

    _draw_zone_kernel_triptych(axes[2, 2], zone_kernel_mats)

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
