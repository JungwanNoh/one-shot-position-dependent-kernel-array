import os
import csv
import numpy as np
from collections import Counter

from config import CFG
from utils import set_seed, ensure_dir
from dataset import load_dataset
from variability import generate_variability_map, generate_observed_image, generate_region_map
from filters import (
    gaussian_kernel,
    apply_global_filter,
    generate_content_map,
    apply_content_aware_pdk,
)
from metrics import compute_all_metrics
from visualize import save_image, save_region_map, save_panel, save_summary_barplot


def prepare_samples(dataset):
    """
    observed / region_map을 한 번만 생성해서
    모든 candidate가 동일한 observed에 대해 평가되도록 고정
    """
    samples = []
    for name, clean in dataset:
        h, w = clean.shape
        v_map = generate_variability_map(h, w, mode=CFG.MAP_MODE)
        observed, noise_sigma_map = generate_observed_image(clean, v_map)
        region_map = generate_region_map(v_map)

        samples.append({
            "name": name,
            "clean": clean,
            "v_map": v_map,
            "observed": observed,
            "noise_sigma_map": noise_sigma_map,
            "region_map": region_map,
        })
    return samples


def split_samples(samples, val_ratio: float):
    n = len(samples)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1) if n > 1 else 1
    return samples[:n_val], samples[n_val:]


def mean_metric_dict(dicts):
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        out[k] = float(np.mean(vals))
    return out


def add_selection_scores(candidate_rows, weights):
    """
    min-max normalization among candidates on validation metrics
    """
    metric_names = ["mse", "grad", "edge_mse", "region_mse"]

    for m in metric_names:
        vals = np.array([row[m] for row in candidate_rows], dtype=np.float64)
        vmin, vmax = vals.min(), vals.max()
        denom = vmax - vmin

        for row in candidate_rows:
            if denom < 1e-12:
                row[f"{m}_norm"] = 0.0
            else:
                row[f"{m}_norm"] = float((row[m] - vmin) / denom)

    for row in candidate_rows:
        score = 0.0
        for m in metric_names:
            score += weights[m] * row[f"{m}_norm"]
        row["selection_score"] = float(score)

    return candidate_rows


def select_best_global_sigma(val_samples):
    rows = []
    for sigma in CFG.GLOBAL_SIGMA_CANDIDATES:
        metrics_all = []
        for s in val_samples:
            pred = apply_global_filter(s["observed"], sigma=sigma, kernel_size=CFG.KERNEL_SIZE)
            metrics = compute_all_metrics(
                pred=pred,
                target=s["clean"],
                region_map=s["region_map"],
                edge_percentile=CFG.EDGE_PERCENTILE_FOR_SCORE,
            )
            metrics_all.append(metrics)

        mean_metrics = mean_metric_dict(metrics_all)
        row = {"sigma": float(sigma), **mean_metrics}
        rows.append(row)

    rows = add_selection_scores(rows, CFG.SCORE_WEIGHTS)
    best_row = min(rows, key=lambda x: x["selection_score"])
    return best_row, rows


def select_best_pdk_preset(val_samples):
    rows = []
    for preset in CFG.RULE_PRESETS:
        metrics_all = []
        for s in val_samples:
            content_map = generate_content_map(s["observed"])
            pred = apply_content_aware_pdk(
                img=s["observed"],
                region_map=s["region_map"],
                content_map=content_map,
                lut=preset["lut"],
                kernel_size=CFG.KERNEL_SIZE,
            )
            metrics = compute_all_metrics(
                pred=pred,
                target=s["clean"],
                region_map=s["region_map"],
                edge_percentile=CFG.EDGE_PERCENTILE_FOR_SCORE,
            )
            metrics_all.append(metrics)

        mean_metrics = mean_metric_dict(metrics_all)
        row = {"preset_name": preset["name"], "preset": preset, **mean_metrics}
        rows.append(row)

    rows = add_selection_scores(rows, CFG.SCORE_WEIGHTS)
    best_row = min(rows, key=lambda x: x["selection_score"])
    return best_row, rows


def print_candidate_report(title, rows, key_name):
    print(f"\n=== {title} ===")
    for row in rows:
        print(
            f"{key_name}={row[key_name]} | "
            f"MSE={row['mse']:.6f} | "
            f"Grad={row['grad']:.6f} | "
            f"EdgeMSE={row['edge_mse']:.6f} | "
            f"RegionMSE={row['region_mse']:.6f} | "
            f"Score={row['selection_score']:.6f}"
        )


def evaluate_on_test(test_samples, selected_global_sigma, selected_preset):
    img_dir = os.path.join(CFG.SAVE_ROOT, "images")
    panel_dir = os.path.join(CFG.SAVE_ROOT, "panels")
    plot_dir = os.path.join(CFG.SAVE_ROOT, "plots")
    ensure_dir(img_dir)
    ensure_dir(panel_dir)
    ensure_dir(plot_dir)

    all_rows = []

    psnr_obs_list, psnr_global_list, psnr_pdk_list = [], [], []
    ssim_obs_list, ssim_global_list, ssim_pdk_list = [], [], []
    grad_obs_list, grad_global_list, grad_pdk_list = [], [], []

    for idx, s in enumerate(test_samples):
        name = s["name"]
        clean = s["clean"]
        observed = s["observed"]
        v_map = s["v_map"]
        region_map = s["region_map"]

        content_map = generate_content_map(observed)

        global_out = apply_global_filter(observed, sigma=selected_global_sigma, kernel_size=CFG.KERNEL_SIZE)
        pdk_out = apply_content_aware_pdk(
            img=observed,
            region_map=region_map,
            content_map=content_map,
            lut=selected_preset["lut"],
            kernel_size=CFG.KERNEL_SIZE,
        )

        met_obs = compute_all_metrics(observed, clean, region_map, edge_percentile=CFG.EDGE_PERCENTILE_FOR_SCORE)
        met_global = compute_all_metrics(global_out, clean, region_map, edge_percentile=CFG.EDGE_PERCENTILE_FOR_SCORE)
        met_pdk = compute_all_metrics(pdk_out, clean, region_map, edge_percentile=CFG.EDGE_PERCENTILE_FOR_SCORE)

        psnr_obs_list.append(met_obs["psnr"])
        psnr_global_list.append(met_global["psnr"])
        psnr_pdk_list.append(met_pdk["psnr"])

        ssim_obs_list.append(met_obs["ssim"])
        ssim_global_list.append(met_global["ssim"])
        ssim_pdk_list.append(met_pdk["ssim"])

        grad_obs_list.append(met_obs["grad"])
        grad_global_list.append(met_global["grad"])
        grad_pdk_list.append(met_pdk["grad"])

        row = {
            "name": name,
            "psnr_observed": met_obs["psnr"],
            "psnr_global": met_global["psnr"],
            "psnr_pdk": met_pdk["psnr"],
            "ssim_observed": met_obs["ssim"],
            "ssim_global": met_global["ssim"],
            "ssim_pdk": met_pdk["ssim"],
            "grad_observed": met_obs["grad"],
            "grad_global": met_global["grad"],
            "grad_pdk": met_pdk["grad"],
            "mse_observed": met_obs["mse"],
            "mse_global": met_global["mse"],
            "mse_pdk": met_pdk["mse"],
            "edge_mse_observed": met_obs["edge_mse"],
            "edge_mse_global": met_global["edge_mse"],
            "edge_mse_pdk": met_pdk["edge_mse"],
            "region_mse_observed": met_obs["region_mse"],
            "region_mse_global": met_global["region_mse"],
            "region_mse_pdk": met_pdk["region_mse"],
        }
        all_rows.append(row)

        if CFG.SAVE_INDIVIDUAL_IMAGES:
            save_image(os.path.join(img_dir, f"{name}_clean.png"), clean)
            save_image(os.path.join(img_dir, f"{name}_observed.png"), observed)
            save_image(os.path.join(img_dir, f"{name}_vmap.png"), v_map, cmap="magma")
            save_region_map(os.path.join(img_dir, f"{name}_regionmap.png"), region_map)
            save_region_map(os.path.join(img_dir, f"{name}_contentmap.png"), content_map)
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
            f"[{idx+1:02d}/{len(test_samples):02d}] {name} | "
            f"PSNR obs/global/pdk = {met_obs['psnr']:.3f} / {met_global['psnr']:.3f} / {met_pdk['psnr']:.3f} | "
            f"Grad obs/global/pdk = {met_obs['grad']:.5f} / {met_global['grad']:.5f} / {met_pdk['grad']:.5f}"
        )

    csv_path = os.path.join(CFG.SAVE_ROOT, "metrics_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_psnr = {
        "Observed": float(np.mean(psnr_obs_list)),
        "Best Global": float(np.mean(psnr_global_list)),
        "Content-aware PDK": float(np.mean(psnr_pdk_list)),
    }
    summary_ssim = {
        "Observed": float(np.mean(ssim_obs_list)),
        "Best Global": float(np.mean(ssim_global_list)),
        "Content-aware PDK": float(np.mean(ssim_pdk_list)),
    }
    summary_grad = {
        "Observed": float(np.mean(grad_obs_list)),
        "Best Global": float(np.mean(grad_global_list)),
        "Content-aware PDK": float(np.mean(grad_pdk_list)),
    }

    if CFG.SAVE_SUMMARY_PLOTS:
        save_summary_barplot(os.path.join(plot_dir, "psnr_summary.png"), summary_psnr, ylabel="PSNR (dB)")
        save_summary_barplot(os.path.join(plot_dir, "ssim_summary.png"), summary_ssim, ylabel="SSIM")
        save_summary_barplot(os.path.join(plot_dir, "gradient_error_summary.png"), summary_grad, ylabel="Gradient Error")

    return summary_psnr, summary_ssim, summary_grad


def main():
    set_seed(CFG.SEED)

    dataset = load_dataset()
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 images for val/test split.")

    samples = prepare_samples(dataset)
    val_samples, test_samples = split_samples(samples, CFG.VAL_RATIO)

    print(f"Total images: {len(samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # ---------------------------------------------------------
    # 1) Select best global sigma using validation composite score
    # ---------------------------------------------------------
    best_global_row, global_rows = select_best_global_sigma(val_samples)
    print_candidate_report("Validation Global Candidates", global_rows, "sigma")

    selected_global_sigma = best_global_row["sigma"]
    selected_global_kernel = gaussian_kernel(CFG.KERNEL_SIZE, selected_global_sigma)

    print("\nSelected best global sigma:")
    print(f"  sigma = {selected_global_sigma:.3f}")
    print(f"  kernel size = {CFG.KERNEL_SIZE}x{CFG.KERNEL_SIZE}")
    print("  kernel =")
    print(np.array2string(selected_global_kernel, precision=4, suppress_small=True))

    # ---------------------------------------------------------
    # 2) Select best content-aware PDK preset on validation
    # ---------------------------------------------------------
    best_pdk_row, pdk_rows = select_best_pdk_preset(val_samples)
    print_candidate_report("Validation PDK Presets", pdk_rows, "preset_name")

    selected_preset = best_pdk_row["preset"]
    print("\nSelected content-aware PDK preset:")
    print(f"  preset = {selected_preset['name']}")
    print("  LUT =")
    for key in sorted(selected_preset["lut"].keys()):
        print(f"    {key}: {selected_preset['lut'][key]}")

    # ---------------------------------------------------------
    # 3) Evaluate on test with fixed selected settings
    # ---------------------------------------------------------
    summary_psnr, summary_ssim, summary_grad = evaluate_on_test(
        test_samples=test_samples,
        selected_global_sigma=selected_global_sigma,
        selected_preset=selected_preset,
    )

    print("\n=== Final Test Summary ===")
    print("PSNR:", summary_psnr)
    print("SSIM:", summary_ssim)
    print("Gradient Error:", summary_grad)
    print(f"Saved to: {CFG.SAVE_ROOT}")


if __name__ == "__main__":
    main()