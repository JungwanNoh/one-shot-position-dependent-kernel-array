import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from config import COMMON, SYNTHETIC
from utils import ensure_dir, list_image_files, load_grayscale, set_seed
from zone_generators import generate_variability_map, observed_from_clean, zone_from_reliability_map
from kernels import apply_global_np, apply_pdk_np
from metrics import compute_metrics


def save_panel(path, clean, observed, v_map, zone_map, global_out, pdk_out):
    ensure_dir(os.path.dirname(path))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(clean, cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("Clean"); axes[0, 0].axis("off")
    axes[0, 1].imshow(observed, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Observed"); axes[0, 1].axis("off")
    axes[0, 2].imshow(v_map, cmap="magma", vmin=0, vmax=1); axes[0, 2].set_title("Reliability"); axes[0, 2].axis("off")
    axes[0, 3].imshow(zone_map, cmap="viridis", vmin=0, vmax=2); axes[0, 3].set_title("PDK Zones"); axes[0, 3].axis("off")
    axes[1, 0].imshow(global_out, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title("Fixed Global"); axes[1, 0].axis("off")
    axes[1, 1].imshow(pdk_out, cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title("PDK"); axes[1, 1].axis("off")
    axes[1, 2].imshow(np.abs(global_out-clean), cmap="inferno"); axes[1, 2].set_title("Err: Global"); axes[1, 2].axis("off")
    axes[1, 3].imshow(np.abs(pdk_out-clean), cmap="inferno"); axes[1, 3].set_title("Err: PDK"); axes[1, 3].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()


def run_synthetic():
    set_seed(42)
    save_dir = SYNTHETIC["save_dir"]
    img_dir = os.path.join(save_dir, "images")
    panel_dir = os.path.join(save_dir, "panels")
    ensure_dir(img_dir); ensure_dir(panel_dir)

    files = list_image_files(SYNTHETIC["root"], max_images=SYNTHETIC["max_images"])
    if not files:
        raise RuntimeError(f"No images found: {SYNTHETIC['root']}")

    rows = []
    all_obs, all_global, all_pdk = [], [], []

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        clean = load_grayscale(path, COMMON["image_size"])
        h, w = clean.shape
        v_map = generate_variability_map(
            h, w,
            mode=SYNTHETIC["map_mode"],
            center=SYNTHETIC["radial_center"],
            falloff=SYNTHETIC["radial_falloff"],
            v_max=SYNTHETIC["v_max"],
            v_min=SYNTHETIC["v_min"],
        )
        rng = np.random.default_rng(42 + idx)
        observed, _ = observed_from_clean(clean, v_map, SYNTHETIC["noise_sigma_min"], SYNTHETIC["noise_sigma_alpha"], rng)
        zone_map = zone_from_reliability_map(v_map, SYNTHETIC["th_high"], SYNTHETIC["th_mid"])

        global_out = apply_global_np(observed, COMMON["global_specs"]["synthetic"], COMMON["kernel_size"])
        pdk_out = apply_pdk_np(observed, zone_map, COMMON["zone_kernel_specs"], COMMON["kernel_size"])

        m_obs = compute_metrics(observed, clean)
        m_global = compute_metrics(global_out, clean)
        m_pdk = compute_metrics(pdk_out, clean)

        rows.append({
            "name": name,
            **{f"{k}_obs": v for k, v in m_obs.items()},
            **{f"{k}_global": v for k, v in m_global.items()},
            **{f"{k}_pdk": v for k, v in m_pdk.items()},
        })
        all_obs.append(m_obs); all_global.append(m_global); all_pdk.append(m_pdk)

        if idx < SYNTHETIC["save_max_panels"]:
            save_panel(os.path.join(panel_dir, f"{name}_panel.png"), clean, observed, v_map, zone_map, global_out, pdk_out)

        print(f"[{idx+1:02d}] {name} | PSNR obs/global/pdk = {m_obs['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}")

    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    def mean_metric(lst, key): return float(np.mean([x[key] for x in lst]))
    print("\n=== Synthetic Summary ===")
    for key in ["psnr", "ssim", "grad"]:
        print(key.upper(), {
            "Observed": mean_metric(all_obs, key),
            "Fixed Global": mean_metric(all_global, key),
            "PDK": mean_metric(all_pdk, key),
        })
