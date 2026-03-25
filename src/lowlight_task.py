import os
import csv
import numpy as np

from config import CFG
from utils import list_image_files, load_grayscale, ensure_dir
from kernels import apply_global_gaussian_np, apply_zonewise_pdk_np
from zone_generators import lowlight_score, quantize_score_to_zone
from metrics import compute_all_metrics
from visualize import save_image, save_zone_map, save_panel, save_barplot


def run():
    save_root = os.path.join(CFG.SAVE_ROOT, "lowlight")
    img_dir = os.path.join(save_root, "images")
    panel_dir = os.path.join(save_root, "panels")
    plot_dir = os.path.join(save_root, "plots")
    ensure_dir(img_dir); ensure_dir(panel_dir); ensure_dir(plot_dir)

    low_files = list_image_files(CFG.LOWLIGHT_INPUT_ROOT, CFG.FILE_EXTENSIONS, CFG.LOWLIGHT_MAX_IMAGES)
    gt_files = list_image_files(CFG.LOWLIGHT_GT_ROOT, CFG.FILE_EXTENSIONS, CFG.LOWLIGHT_MAX_IMAGES)

    low_map = {os.path.splitext(os.path.basename(p))[0]: p for p in low_files}
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}
    names = sorted(list(set(low_map.keys()) & set(gt_map.keys())))

    if not names:
        raise RuntimeError("No paired low/high images found.")

    rows = []
    psnr_raw, psnr_global, psnr_pdk = [], [], []
    ssim_raw, ssim_global, ssim_pdk = [], [], []
    grad_raw, grad_global, grad_pdk = [], [], []

    for idx, name in enumerate(names):
        observed = load_grayscale(low_map[name], CFG.IMAGE_SIZE)
        gt = load_grayscale(gt_map[name], CFG.IMAGE_SIZE)

        global_out = apply_global_gaussian_np(observed, CFG.GLOBAL_SIGMA_LOWLIGHT)
        score = lowlight_score(observed, global_out, CFG.LOWLIGHT_ILLUM_SIGMA, CFG.LOWLIGHT_SCORE_WEIGHTS)
        zone = quantize_score_to_zone(score)
        pdk_out = apply_zonewise_pdk_np(observed, zone, CFG.ZONE_KERNEL_SPECS)

        m_raw = compute_all_metrics(observed, gt)
        m_global = compute_all_metrics(global_out, gt)
        m_pdk = compute_all_metrics(pdk_out, gt)

        rows.append({
            "name": name,
            "psnr_raw": m_raw["psnr"],
            "psnr_global": m_global["psnr"],
            "psnr_pdk": m_pdk["psnr"],
            "ssim_raw": m_raw["ssim"],
            "ssim_global": m_global["ssim"],
            "ssim_pdk": m_pdk["ssim"],
            "grad_raw": m_raw["grad"],
            "grad_global": m_global["grad"],
            "grad_pdk": m_pdk["grad"],
        })

        psnr_raw.append(m_raw["psnr"]); psnr_global.append(m_global["psnr"]); psnr_pdk.append(m_pdk["psnr"])
        ssim_raw.append(m_raw["ssim"]); ssim_global.append(m_global["ssim"]); ssim_pdk.append(m_pdk["ssim"])
        grad_raw.append(m_raw["grad"]); grad_global.append(m_global["grad"]); grad_pdk.append(m_pdk["grad"])

        save_image(os.path.join(img_dir, f"{name}_raw.png"), observed)
        save_image(os.path.join(img_dir, f"{name}_gt.png"), gt)
        save_image(os.path.join(img_dir, f"{name}_score.png"), score, cmap="magma")
        save_zone_map(os.path.join(img_dir, f"{name}_zone.png"), zone)
        save_image(os.path.join(img_dir, f"{name}_global.png"), global_out)
        save_image(os.path.join(img_dir, f"{name}_pdk.png"), pdk_out)

        if idx < CFG.SAVE_MAX_PANELS:
            save_panel(os.path.join(panel_dir, f"{name}_panel.png"), observed, global_out, pdk_out, score, zone, gt)

        print(f"[{idx+1:02d}] {name} | PSNR raw/global/pdk = {m_raw['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}")

    with open(os.path.join(save_root, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_barplot(os.path.join(plot_dir, "psnr.png"), {
        "Raw": float(np.mean(psnr_raw)),
        "Global": float(np.mean(psnr_global)),
        "PDK": float(np.mean(psnr_pdk)),
    }, "PSNR (dB)")
    save_barplot(os.path.join(plot_dir, "ssim.png"), {
        "Raw": float(np.mean(ssim_raw)),
        "Global": float(np.mean(ssim_global)),
        "PDK": float(np.mean(ssim_pdk)),
    }, "SSIM")
    save_barplot(os.path.join(plot_dir, "grad.png"), {
        "Raw": float(np.mean(grad_raw)),
        "Global": float(np.mean(grad_global)),
        "PDK": float(np.mean(grad_pdk)),
    }, "Gradient L1")