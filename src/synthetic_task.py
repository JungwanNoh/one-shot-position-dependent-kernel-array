import os
import csv
import numpy as np
from config import SAVE_ROOT, FILE_EXTENSIONS, GLOBAL_SIGMA, SYNTHETIC, ZONE_KERNEL_SPECS, SEED
from utils import list_image_files, load_grayscale, ensure_dir, clip01
from kernels import apply_global_gaussian_np, apply_zonewise_pdk_np
from zone_generators import synthetic_variability_map, global_first_score_np, clustered_zone_from_score_np
from metrics import compute_all_metrics
from visualize import save_image, save_zone_map, save_panel, save_barplot, save_heatmap


def run():
    save_root = os.path.join(SAVE_ROOT, 'synthetic')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    plot_dir = os.path.join(save_root, 'plots')
    ensure_dir(img_dir)
    ensure_dir(panel_dir)
    ensure_dir(plot_dir)

    files = list_image_files(SYNTHETIC['root'], FILE_EXTENSIONS, SYNTHETIC['max_images'])
    if not files:
        raise RuntimeError(f"No images found: {SYNTHETIC['root']}")

    rows = []
    psnr_obs, psnr_global, psnr_pdk = [], [], []
    ssim_obs, ssim_global, ssim_pdk = [], [], []
    grad_obs, grad_global, grad_pdk = [], [], []
    rng = np.random.default_rng(SEED)

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        clean = load_grayscale(path, SYNTHETIC['image_size'])
        v_map = synthetic_variability_map(clean.shape[0], clean.shape[1])
        sigma_map = SYNTHETIC['noise_sigma_min'] + SYNTHETIC['noise_sigma_alpha'] * (1.0 - v_map)
        noise = rng.normal(0.0, sigma_map, size=clean.shape).astype(np.float32)
        observed = clip01(v_map * clean + noise)
        global_out = apply_global_gaussian_np(observed, GLOBAL_SIGMA['synthetic'])

        w = SYNTHETIC['score_weights']
        score = global_first_score_np(observed, global_out, w['residual'], w['gradient'], (1.0 - v_map), w['extra'])
        zone = clustered_zone_from_score_np(score)
        pdk_out = apply_zonewise_pdk_np(observed, zone, ZONE_KERNEL_SPECS)
        improvement = np.abs(global_out - clean) - np.abs(pdk_out - clean)

        m_obs = compute_all_metrics(observed, clean)
        m_global = compute_all_metrics(global_out, clean)
        m_pdk = compute_all_metrics(pdk_out, clean)
        rows.append({
            'name': name,
            'psnr_observed': m_obs['psnr'],
            'psnr_global': m_global['psnr'],
            'psnr_pdk': m_pdk['psnr'],
            'ssim_observed': m_obs['ssim'],
            'ssim_global': m_global['ssim'],
            'ssim_pdk': m_pdk['ssim'],
            'grad_observed': m_obs['grad'],
            'grad_global': m_global['grad'],
            'grad_pdk': m_pdk['grad'],
        })
        psnr_obs.append(m_obs['psnr'])
        psnr_global.append(m_global['psnr'])
        psnr_pdk.append(m_pdk['psnr'])
        ssim_obs.append(m_obs['ssim'])
        ssim_global.append(m_global['ssim'])
        ssim_pdk.append(m_pdk['ssim'])
        grad_obs.append(m_obs['grad'])
        grad_global.append(m_global['grad'])
        grad_pdk.append(m_pdk['grad'])

        save_image(os.path.join(img_dir, f'{name}_clean.png'), clean)
        save_image(os.path.join(img_dir, f'{name}_obs.png'), observed)
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
                clean,
                aux_title='Improvement',
                aux_kind='improvement',
            )
        print(f"[{idx+1:02d}] {name} | PSNR obs/global/pdk = {m_obs['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}")

    with open(os.path.join(save_root, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_barplot(os.path.join(plot_dir, 'psnr.png'), {'Observed': float(np.mean(psnr_obs)), 'Global': float(np.mean(psnr_global)), 'PDK': float(np.mean(psnr_pdk))}, 'PSNR (dB)')
    save_barplot(os.path.join(plot_dir, 'ssim.png'), {'Observed': float(np.mean(ssim_obs)), 'Global': float(np.mean(ssim_global)), 'PDK': float(np.mean(ssim_pdk))}, 'SSIM')
    save_barplot(os.path.join(plot_dir, 'grad.png'), {'Observed': float(np.mean(grad_obs)), 'Global': float(np.mean(grad_global)), 'PDK': float(np.mean(grad_pdk))}, 'Gradient L1')
