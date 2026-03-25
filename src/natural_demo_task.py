import os
import matplotlib.pyplot as plt

from config import COMMON, NATURAL
from utils import ensure_dir, list_image_files, load_grayscale, set_seed
from zone_generators import edge_energy_center, foveated_zone_from_center
from kernels import apply_global_np, apply_pdk_np


def save_panel(path, img, energy, zone_map, global_out, pdk_out):
    ensure_dir(os.path.dirname(path))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(energy, cmap="magma", vmin=0, vmax=1); axes[1].set_title("Pseudo-ROI Energy"); axes[1].axis("off")
    axes[2].imshow(zone_map, cmap="viridis", vmin=0, vmax=2); axes[2].set_title("Foveated Zones"); axes[2].axis("off")
    axes[3].imshow(global_out, cmap="gray", vmin=0, vmax=1); axes[3].set_title("Fixed Global"); axes[3].axis("off")
    axes[4].imshow(pdk_out, cmap="gray", vmin=0, vmax=1); axes[4].set_title("PDK"); axes[4].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()


def run_natural_demo():
    set_seed(42)
    save_dir = NATURAL["save_dir"]
    panel_dir = os.path.join(save_dir, "panels")
    ensure_dir(panel_dir)

    files = list_image_files(NATURAL["root"], max_images=NATURAL["max_images"])
    if not files:
        raise RuntimeError(f"No images found: {NATURAL['root']}")

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_grayscale(path, COMMON["image_size"])
        cx, cy, energy = edge_energy_center(img)
        _, zone_map = foveated_zone_from_center(img.shape[0], img.shape[1], cx, cy, NATURAL["r1"], NATURAL["r2"])

        global_out = apply_global_np(img, COMMON["global_specs"]["natural"], COMMON["kernel_size"])
        pdk_out = apply_pdk_np(img, zone_map, COMMON["zone_kernel_specs"], COMMON["kernel_size"])

        if idx < NATURAL["save_max_panels"]:
            save_panel(os.path.join(panel_dir, f"{name}_panel.png"), img, energy, zone_map, global_out, pdk_out)

        print(f"[{idx+1:02d}] {name} | pseudo-center=({cx:.3f},{cy:.3f})")

    print(f"\nNatural demo panels saved to: {save_dir}")
