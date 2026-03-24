import os
import csv
import numpy as np
from collections import Counter

from config import CFG
from utils import set_seed, ensure_dir
from dataset import build_samples
from gaze_zone import build_zone_maps
from filters import apply_global_gaussian, apply_zonewise_pdk, gaussian_kernel
from metrics import compute_all_metrics, selection_score
from visualize import save_image, save_zone_map, save_panel, save_barplot


def evaluate_candidate_global(samples, sigma: float):
    scores = []
    for s in samples:
        pred = apply_global_gaussian(s["observed"], sigma=sigma)
        scores.append(selection_score(pred, s["clean"]))
    return float(np.mean(scores))


def evaluate_candidate_preset(samples, preset: dict):
    scores = []
    for s in samples:
        gaze_map, base_zone, final_zone = build_zone_maps(s["gaze_xy"], s["v_map"])
        pred = apply_zonewise_pdk(s["observed"], final_zone, preset["zone_specs"])
        scores.append(selection_score(pred, s["clean"]))
    return float(np.mean(scores))


def select_best_global_sigma(val_samples):
    rows = []
    for sigma in CFG.GLOBAL_SIGMA_CANDIDATES:
        score = evaluate_candidate_global(val_samples, sigma)
        rows.append({"sigma": float(sigma), "score": score})
    best_row = min(rows, key=lambda x: x["score"])
    return best_row, rows


def select_best_preset(val_samples):
    rows = []
    for preset in CFG.PDK_PRESETS:
        score = evaluate_candidate_preset(val_samples, preset)
        rows.append({"name": preset["name"], "preset": preset, "score": score})
    best_row = min(rows, key=lambda x: x["score"])
    return best_row, rows


def evaluate_test(test_samples, best_sigma: float, best_preset: dict):
    img_dir = os.path.join(CFG.SAVE_ROOT, "images")
    panel_dir = os.path.join(CFG.SAVE_ROOT, "panels")
    plot_dir = os.path.join(CFG.SAVE_ROOT, "plots")
    ensure_dir(img_dir)
    ensure_dir(panel_dir)
    ensure_dir(plot_dir)

    rows = []

    psnr_obs, psnr_global, psnr_pdk = [], [], []
    ssim_obs, ssim_global, ssim_pdk = [], [], []
    grad_obs, grad_global, grad_pdk = [], [], []

    for idx, s in enumerate(test_samples):
        clean = s["clean"]
        observed = s["observed"]
        v_map = s["v_map"]
        gaze_xy = s["gaze_xy"]
        name = s["name"]

        gaze_map, base_zone, final_zone = build_zone_maps(gaze_xy, v_map)

        out_global = apply_global_gaussian(observed, sigma=best_sigma)
        out_pdk = apply_zonewise_pdk(observed, final_zone, best_preset["zone_specs"])

        m_obs = compute_all_metrics(observed, clean)
        m_global = compute_all_metrics(out_global, clean)
        m_pdk = compute_all_metrics(out_pdk, clean)

        rows.append({
            "name": name,
            "gaze_x": gaze_xy[0],
            "gaze_y": gaze_xy[1],
            "psnr_observed": m_obs["psnr"],
            "psnr_global": m_global["psnr"],
            "psnr_pdk": m_pdk["psnr"],
            "ssim_observed": m_obs["ssim"],
            "ssim_global": m_global["ssim"],
            "ssim_pdk": m_pdk["ssim"],
            "grad_observed": m_obs["grad"],
            "grad_global": m_global["grad"],
            "grad_pdk": m_pdk["grad"],
            "l1_observed": m_obs["l1"],
            "l1_global": m_global["l1"],
            "l1_pdk": m_pdk["l1"],
            "mse_observed": m_obs["mse"],
            "mse_global": m_global["mse"],
            "mse_pdk": m_pdk["mse"],
        })

        psnr_obs.append(m_obs["psnr"])
        psnr_global.append(m_global["psnr"])
        psnr_pdk.append(m_pdk["psnr"])

        ssim_obs.append(m_obs["ssim"])
        ssim_global.append(m_global["ssim"])
        ssim_pdk.append(m_pdk["ssim"])

        grad_obs.append(m_obs["grad"])
        grad_global.append(m_global["grad"])
        grad_pdk.append(m_pdk["grad"])

        if CFG.SAVE_INDIVIDUAL_IMAGES:
            save_image(os.path.join(img_dir, f"{name}_clean.png"), clean)
            save_image(os.path.join(img_dir, f"{name}_observed.png"), observed)
            save_image(os.path.join(img_dir, f"{name}_gaze_map.png"), gaze_map, cmap="magma")
            save_zone_map(os.path.join(img_dir, f"{name}_base_zone.png"), base_zone)
            save_zone_map(os.path.join(img_dir, f"{name}_final_zone.png"), final_zone)
            save_image(os.path.join(img_dir, f"{name}_global.png"), out_global)
            save_image(os.path.join(img_dir, f"{name}_pdk.png"), out_pdk)

        if CFG.SAVE_PANELS and idx < CFG.SAVE_MAX_PANELS:
            save_panel(
                os.path.join(panel_dir, f"{name}_panel.png"),
                clean=clean,
                observed=observed,
                global_out=out_global,
                pdk_out=out_pdk,
                gaze_map=gaze_map,
                base_zone=base_zone,
                final_zone=final_zone,
            )

        print(
            f"[{idx+1:02d}] {name} | gaze=({gaze_xy[0]:.3f},{gaze_xy[1]:.3f}) | "
            f"PSNR obs/global/pdk = {m_obs['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f} | "
            f"Grad obs/global/pdk = {m_obs['grad']:.5f} / {m_global['grad']:.5f} / {m_pdk['grad']:.5f}"
        )

    csv_path = os.path.join(CFG.SAVE_ROOT, "metrics_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_psnr = {
        "Observed": float(np.mean(psnr_obs)),
        "Best Global": float(np.mean(psnr_global)),
        "Gaze/Sensor PDK": float(np.mean(psnr_pdk)),
    }
    summary_ssim = {
        "Observed": float(np.mean(ssim_obs)),
        "Best Global": float(np.mean(ssim_global)),
        "Gaze/Sensor PDK": float(np.mean(ssim_pdk)),
    }
    summary_grad = {
        "Observed": float(np.mean(grad_obs)),
        "Best Global": float(np.mean(grad_global)),
        "Gaze/Sensor PDK": float(np.mean(grad_pdk)),
    }

    save_barplot(os.path.join(plot_dir, "psnr_summary.png"), summary_psnr, ylabel="PSNR (dB)")
    save_barplot(os.path.join(plot_dir, "ssim_summary.png"), summary_ssim, ylabel="SSIM")
    save_barplot(os.path.join(plot_dir, "gradient_summary.png"), summary_grad, ylabel="Gradient L1")

    return summary_psnr, summary_ssim, summary_grad


def main():
    set_seed(CFG.SEED)

    val_samples = build_samples(CFG.VAL_ROOT, CFG.MAX_VAL_IMAGES, base_seed=CFG.SEED * 10 + 1)
    test_samples = build_samples(CFG.TEST_ROOT, CFG.MAX_TEST_IMAGES, base_seed=CFG.SEED * 10 + 2)

    if len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Validation or test dataset is empty.")

    best_global_row, global_rows = select_best_global_sigma(val_samples)
    best_sigma = best_global_row["sigma"]
    best_kernel = gaussian_kernel(CFG.KERNEL_SIZE, best_sigma)

    print("\n=== Validation Global Search ===")
    for row in global_rows:
        print(f"sigma={row['sigma']:.2f} | score={row['score']:.6f}")

    print("\nSelected best global sigma:")
    print(f"  sigma = {best_sigma:.3f}")
    print(f"  kernel size = {CFG.KERNEL_SIZE}x{CFG.KERNEL_SIZE}")
    print("  kernel =")
    print(np.array2string(best_kernel, precision=4, suppress_small=True))

    best_preset_row, preset_rows = select_best_preset(val_samples)
    best_preset = best_preset_row["preset"]

    print("\n=== Validation PDK Preset Search ===")
    for row in preset_rows:
        print(f"preset={row['name']} | score={row['score']:.6f}")

    print("\nSelected PDK preset:")
    print(f"  name = {best_preset['name']}")
    for zone_id in [0, 1, 2]:
        print(f"  zone {zone_id}: {best_preset['zone_specs'][zone_id]}")

    summary_psnr, summary_ssim, summary_grad = evaluate_test(
        test_samples=test_samples,
        best_sigma=best_sigma,
        best_preset=best_preset,
    )

    print("\n=== Final Test Summary ===")
    print("PSNR:", summary_psnr)
    print("SSIM:", summary_ssim)
    print("Gradient:", summary_grad)
    print(f"Saved to: {CFG.SAVE_ROOT}")


if __name__ == "__main__":
    main()