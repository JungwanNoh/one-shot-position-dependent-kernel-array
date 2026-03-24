import os
import csv
import numpy as np

from config import CFG
from utils import set_seed, ensure_dir
from dataset import load_dataset
from variability import generate_variability_map, generate_observed_image, generate_region_map
from collections import Counter
from filters import apply_pdk_filter, search_best_global_sigma, gaussian_kernel
from metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_regionwise_mse,
    compute_gradient_error,
)
from visualize import save_image, save_region_map, save_panel, save_summary_barplot


def main():
    set_seed(CFG.SEED)

    img_dir = os.path.join(CFG.SAVE_ROOT, "images")
    panel_dir = os.path.join(CFG.SAVE_ROOT, "panels")
    plot_dir = os.path.join(CFG.SAVE_ROOT, "plots")
    ensure_dir(img_dir)
    ensure_dir(panel_dir)
    ensure_dir(plot_dir)

    dataset = load_dataset()
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in {CFG.DATA_ROOT}")

    all_rows = []

    psnr_obs_list, psnr_global_list, psnr_pdk_list = [], [], []
    ssim_obs_list, ssim_global_list, ssim_pdk_list = [], [], []
    grad_obs_list, grad_global_list, grad_pdk_list = [], [], []

    region_global_accum = {0: [], 1: [], 2: []}
    region_pdk_accum = {0: [], 1: [], 2: []}
    region_obs_accum = {0: [], 1: [], 2: []}

    best_sigma_list = []

    for idx, (name, clean) in enumerate(dataset):
        h, w = clean.shape

        v_map = generate_variability_map(h, w, mode=CFG.MAP_MODE)
        observed, noise_sigma_map = generate_observed_image(clean, v_map)
        region_map = generate_region_map(v_map)

        best_sigma, global_out, _ = search_best_global_sigma(
            observed=observed,
            clean=clean,
            sigma_candidates=list(CFG.GLOBAL_SIGMA_CANDIDATES),
            kernel_size=CFG.KERNEL_SIZE,
        )
        pdk_out = apply_pdk_filter(observed, region_map)

        mse_obs = compute_mse(observed, clean)
        mse_global = compute_mse(global_out, clean)
        mse_pdk = compute_mse(pdk_out, clean)

        psnr_obs = compute_psnr(observed, clean)
        psnr_global = compute_psnr(global_out, clean)
        psnr_pdk = compute_psnr(pdk_out, clean)

        ssim_obs = compute_ssim(observed, clean)
        ssim_global = compute_ssim(global_out, clean)
        ssim_pdk = compute_ssim(pdk_out, clean)

        grad_obs = compute_gradient_error(observed, clean)
        grad_global = compute_gradient_error(global_out, clean)
        grad_pdk = compute_gradient_error(pdk_out, clean)

        reg_obs = compute_regionwise_mse(observed, clean, region_map)
        reg_global = compute_regionwise_mse(global_out, clean, region_map)
        reg_pdk = compute_regionwise_mse(pdk_out, clean, region_map)

        best_sigma_list.append(best_sigma)

        psnr_obs_list.append(psnr_obs)
        psnr_global_list.append(psnr_global)
        psnr_pdk_list.append(psnr_pdk)

        ssim_obs_list.append(ssim_obs)
        ssim_global_list.append(ssim_global)
        ssim_pdk_list.append(ssim_pdk)

        grad_obs_list.append(grad_obs)
        grad_global_list.append(grad_global)
        grad_pdk_list.append(grad_pdk)

        for rid in [0, 1, 2]:
            region_obs_accum[rid].append(reg_obs[rid])
            region_global_accum[rid].append(reg_global[rid])
            region_pdk_accum[rid].append(reg_pdk[rid])

        row = {
            "name": name,
            "best_global_sigma": best_sigma,
            "mse_observed": mse_obs,
            "mse_global": mse_global,
            "mse_pdk": mse_pdk,
            "psnr_observed": psnr_obs,
            "psnr_global": psnr_global,
            "psnr_pdk": psnr_pdk,
            "ssim_observed": ssim_obs,
            "ssim_global": ssim_global,
            "ssim_pdk": ssim_pdk,
            "grad_observed": grad_obs,
            "grad_global": grad_global,
            "grad_pdk": grad_pdk,
            "region0_obs": reg_obs[0],
            "region1_obs": reg_obs[1],
            "region2_obs": reg_obs[2],
            "region0_global": reg_global[0],
            "region1_global": reg_global[1],
            "region2_global": reg_global[2],
            "region0_pdk": reg_pdk[0],
            "region1_pdk": reg_pdk[1],
            "region2_pdk": reg_pdk[2],
        }
        all_rows.append(row)

        if CFG.SAVE_INDIVIDUAL_IMAGES:
            save_image(os.path.join(img_dir, f"{name}_clean.png"), clean)
            save_image(os.path.join(img_dir, f"{name}_observed.png"), observed)
            save_image(os.path.join(img_dir, f"{name}_vmap.png"), v_map, cmap="magma")
            save_region_map(os.path.join(img_dir, f"{name}_regionmap.png"), region_map)
            save_image(os.path.join(img_dir, f"{name}_global.png"), global_out)
            save_image(os.path.join(img_dir, f"{name}_pdk.png"), pdk_out)

        if CFG.SAVE_PANELS and idx < 8:
            save_panel(
                os.path.join(panel_dir, f"{name}_panel.png"),
                clean=clean,
                observed=observed,
                global_out=global_out,
                pdk_out=pdk_out,
                v_map=v_map,
                region_map=region_map,
            )

        print(
            f"[{idx+1:02d}/{len(dataset):02d}] {name} | "
            f"best global sigma={best_sigma:.2f} | "
            f"PSNR obs/global/pdk = {psnr_obs:.3f} / {psnr_global:.3f} / {psnr_pdk:.3f}"
        )

    csv_path = os.path.join(CFG.SAVE_ROOT, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_psnr = {
        "Observed": float(np.mean(psnr_obs_list)),
        "Best Global": float(np.mean(psnr_global_list)),
        "Ideal PDK": float(np.mean(psnr_pdk_list)),
    }
    summary_ssim = {
        "Observed": float(np.mean(ssim_obs_list)),
        "Best Global": float(np.mean(ssim_global_list)),
        "Ideal PDK": float(np.mean(ssim_pdk_list)),
    }
    summary_grad = {
        "Observed": float(np.mean(grad_obs_list)),
        "Best Global": float(np.mean(grad_global_list)),
        "Ideal PDK": float(np.mean(grad_pdk_list)),
    }

    if CFG.SAVE_SUMMARY_PLOTS:
        save_summary_barplot(os.path.join(plot_dir, "psnr_summary.png"), summary_psnr, ylabel="PSNR (dB)")
        save_summary_barplot(os.path.join(plot_dir, "ssim_summary.png"), summary_ssim, ylabel="SSIM")
        save_summary_barplot(os.path.join(plot_dir, "gradient_error_summary.png"), summary_grad, ylabel="Gradient Error")

    sigma_counter = Counter(best_sigma_list)
    most_common_sigma, count = sigma_counter.most_common(1)[0]
    most_common_kernel = gaussian_kernel(CFG.KERNEL_SIZE, most_common_sigma)

    print("\n=== Summary ===")
    print("PSNR:", summary_psnr)
    print("SSIM:", summary_ssim)
    print("Gradient Error:", summary_grad)
    print(f"Most common best global sigma: {most_common_sigma:.3f} (selected {count} times)")
    print(f"Kernel size: {CFG.KERNEL_SIZE}x{CFG.KERNEL_SIZE}")
    print("Gaussian kernel for most common best sigma:")
    print(np.array2string(most_common_kernel, precision=4, suppress_small=True))
    print(f"Saved to: {CFG.SAVE_ROOT}")


if __name__ == "__main__":
    main()