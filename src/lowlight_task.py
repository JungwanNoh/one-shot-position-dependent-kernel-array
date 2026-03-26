import os
import csv
import numpy as np
from scipy.ndimage import gaussian_filter
from config import SAVE_ROOT, FILE_EXTENSIONS, GLOBAL_SIGMA, LOWLIGHT, ZONE_KERNEL_SPECS
from utils import list_image_files, load_grayscale, ensure_dir, normalize01
from kernels import apply_global_gaussian_np, apply_zonewise_pdk_np
from zone_generators import global_first_score_np, clustered_zone_from_score_np
from metrics import compute_all_metrics
from visualize import save_image, save_zone_map, save_panel, save_barplot, save_heatmap


def run():
    save_root = os.path.join(SAVE_ROOT, 'lowlight')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    plot_dir = os.path.join(save_root, 'plots')
    ensure_dir(img_dir)
    ensure_dir(panel_dir)
    ensure_dir(plot_dir)

    low_files = list_image_files(LOWLIGHT['input_root'], FILE_EXTENSIONS, LOWLIGHT['max_images'])
    gt_files = list_image_files(LOWLIGHT['gt_root'], FILE_EXTENSIONS, LOWLIGHT['max_images'])
    low_map = {os.path.splitext(os.path.basename(p))[0]: p for p in low_files}
    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_files}
    names = sorted(list(set(low_map.keys()) & set(gt_map.keys())))
    if not names:
        raise RuntimeError('No paired low/high images found. Check LOWLIGHT paths in config.py')

    rows = []
    psnr_raw, psnr_global, psnr_pdk = [], [], []
    ssim_raw, ssim_global, ssim_pdk = [], [], []
    grad_raw, grad_global, grad_pdk = [], [], []

    for idx, name in enumerate(names):
        observed = load_grayscale(low_map[name], LOWLIGHT['image_size'])
        gt = load_grayscale(gt_map[name], LOWLIGHT['image_size'])
        global_out = apply_global_gaussian_np(observed, GLOBAL_SIGMA['lowlight'])

        illum = gaussian_filter(observed, sigma=LOWLIGHT['illum_sigma'])
        lowlightness = 1.0 - normalize01(illum)
        w = LOWLIGHT['score_weights']
        score = global_first_score_np(observed, global_out, w['residual'], w['gradient'], lowlightness, w['extra'])
        zone = clustered_zone_from_score_np(score)
        pdk_out = apply_zonewise_pdk_np(observed, zone, ZONE_KERNEL_SPECS)
        improvement = np.abs(global_out - gt) - np.abs(pdk_out - gt)

        m_raw = compute_all_metrics(observed, gt)
        m_global = compute_all_metrics(global_out, gt)
        m_pdk = compute_all_metrics(pdk_out, gt)
        rows.append({
            'name': name,
            'psnr_raw': m_raw['psnr'],
            'psnr_global': m_global['psnr'],
            'psnr_pdk': m_pdk['psnr'],
            'ssim_raw': m_raw['ssim'],
            'ssim_global': m_global['ssim'],
            'ssim_pdk': m_pdk['ssim'],
            'grad_raw': m_raw['grad'],
            'grad_global': m_global['grad'],
            'grad_pdk': m_pdk['grad'],
        })
        psnr_raw.append(m_raw['psnr'])
        psnr_global.append(m_global['psnr'])
        psnr_pdk.append(m_pdk['psnr'])
        ssim_raw.append(m_raw['ssim'])
        ssim_global.append(m_global['ssim'])
        ssim_pdk.append(m_pdk['ssim'])
        grad_raw.append(m_raw['grad'])
        grad_global.append(m_global['grad'])
        grad_pdk.append(m_pdk['grad'])

        save_image(os.path.join(img_dir, f'{name}_raw.png'), observed)
        save_image(os.path.join(img_dir, f'{name}_gt.png'), gt)
        save_heatmap(os.path.join(img_dir, f'{name}_score.png'), score, cmap='magma', vmin=0.0, vmax=1.0)
        save_zone_map(os.path.join(img_dir, f'{name}_zone.png'), zone)
        save_image(os.path.join(img_dir, f'{name}_global.png'), global_out)
        save_image(os.path.join(img_dir, f'{name}_pdk.png'), pdk_out)
        save_heatmap(os.path.join(img_dir, f'{name}_improvement.png'), improvement, cmap='coolwarm')

        if idx < 8:
            save_panel(
                os.path.join(panel_dir, f'{name}_panel.png'),
                observed,
                global_out,
                pdk_out,
                improvement,
                zone,
                gt,
                aux_title='Improvement',
                aux_kind='improvement',
            )
        print(f"[{idx+1:02d}] {name} | PSNR raw/global/pdk = {m_raw['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}")

    with open(os.path.join(save_root, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_barplot(os.path.join(plot_dir, 'psnr.png'), {'Raw': float(sum(psnr_raw) / len(psnr_raw)), 'Global': float(sum(psnr_global) / len(psnr_global)), 'PDK': float(sum(psnr_pdk) / len(psnr_pdk))}, 'PSNR (dB)')
    save_barplot(os.path.join(plot_dir, 'ssim.png'), {'Raw': float(sum(ssim_raw) / len(ssim_raw)), 'Global': float(sum(ssim_global) / len(ssim_global)), 'PDK': float(sum(ssim_pdk) / len(ssim_pdk))}, 'SSIM')
    save_barplot(os.path.join(plot_dir, 'grad.png'), {'Raw': float(sum(grad_raw) / len(grad_raw)), 'Global': float(sum(grad_global) / len(grad_global)), 'PDK': float(sum(grad_pdk) / len(grad_pdk))}, 'Gradient L1')
