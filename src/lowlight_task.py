import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import COMMON, LOWLIGHT
from utils import ensure_dir, load_grayscale, set_seed
from zone_generators import zone_from_illumination
from kernels import apply_global_np, apply_pdk_np
from metrics import compute_metrics


def matched_pairs(low_dir, high_dir, max_images=None):
    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        return []
    low_files = {os.path.basename(p): os.path.join(low_dir, p) for p in os.listdir(low_dir)}
    high_files = {os.path.basename(p): os.path.join(high_dir, p) for p in os.listdir(high_dir)}
    names = sorted(set(low_files) & set(high_files))
    if max_images is not None:
        names = names[:max_images]
    return [(low_files[n], high_files[n]) for n in names]


def save_panel(path, target, low, illum, zone_map, global_out, pdk_out):
    ensure_dir(os.path.dirname(path))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(target, cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("Target"); axes[0, 0].axis("off")
    axes[0, 1].imshow(low, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Low-light"); axes[0, 1].axis("off")
    axes[0, 2].imshow(illum, cmap="magma", vmin=0, vmax=1); axes[0, 2].set_title("Illumination"); axes[0, 2].axis("off")
    axes[0, 3].imshow(zone_map, cmap="viridis", vmin=0, vmax=2); axes[0, 3].set_title("PDK Zones"); axes[0, 3].axis("off")
    axes[1, 0].imshow(global_out, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title("Fixed Global"); axes[1, 0].axis("off")
    axes[1, 1].imshow(pdk_out, cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title("PDK"); axes[1, 1].axis("off")
    axes[1, 2].imshow(np.abs(global_out-target), cmap="inferno"); axes[1, 2].set_title("Err: Global"); axes[1, 2].axis("off")
    axes[1, 3].imshow(np.abs(pdk_out-target), cmap="inferno"); axes[1, 3].set_title("Err: PDK"); axes[1, 3].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()


def run_lowlight():
    set_seed(42)
    save_dir = LOWLIGHT["save_dir"]
    panel_dir = os.path.join(save_dir, "panels")
    ensure_dir(panel_dir)

    pairs = matched_pairs(LOWLIGHT["low_dir"], LOWLIGHT["high_dir"], LOWLIGHT["max_images"])
    if not pairs:
        raise RuntimeError(f"No matched low/high image pairs found in {LOWLIGHT['low_dir']} and {LOWLIGHT['high_dir']}")

    rows = []
    all_obs, all_global, all_pdk = [], [], []

    for idx, (low_path, high_path) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(low_path))[0]
        low = load_grayscale(low_path, COMMON["image_size"])
        target = load_grayscale(high_path, COMMON["image_size"])
        illum, zone_map = zone_from_illumination(low, LOWLIGHT["illum_high"], LOWLIGHT["illum_mid"])

        global_out = apply_global_np(low, COMMON["global_specs"]["lowlight"], COMMON["kernel_size"])
        pdk_out = apply_pdk_np(low, zone_map, COMMON["zone_kernel_specs"], COMMON["kernel_size"])

        m_obs = compute_metrics(low, target)
        m_global = compute_metrics(global_out, target)
        m_pdk = compute_metrics(pdk_out, target)
        rows.append({
            "name": name,
            **{f"{k}_obs": v for k, v in m_obs.items()},
            **{f"{k}_global": v for k, v in m_global.items()},
            **{f"{k}_pdk": v for k, v in m_pdk.items()},
        })
        all_obs.append(m_obs); all_global.append(m_global); all_pdk.append(m_pdk)

        if idx < LOWLIGHT["save_max_panels"]:
            save_panel(os.path.join(panel_dir, f"{name}_panel.png"), target, low, illum, zone_map, global_out, pdk_out)

        print(f"[{idx+1:02d}] {name} | PSNR low/global/pdk = {m_obs['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}")

    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    def mean_metric(lst, key): return float(np.mean([x[key] for x in lst]))
    print("\n=== Low-light Summary ===")
    for key in ["psnr", "ssim", "grad"]:
        print(key.upper(), {
            "Observed": mean_metric(all_obs, key),
            "Fixed Global": mean_metric(all_global, key),
            "PDK": mean_metric(all_pdk, key),
        })
