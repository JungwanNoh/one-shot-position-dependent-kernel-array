import os
import csv
import numpy as np
from scipy.ndimage import gaussian_filter

from config import SAVE_ROOT, FILE_EXTENSIONS, SYNTHETIC, GLOBAL_KERNEL_SPECS, SEED
from utils import list_image_files, load_grayscale, ensure_dir, clip01
from kernels import (
    apply_global_kernel_np,
    apply_zonewise_pdk_np,
    format_kernel_spec,
    format_zone_kernel_specs,
    get_global_kernel_matrix,
    get_zone_kernel_matrices,
)
from zone_generators import (
    random_field_map,
    synthetic_scores_np,
    assign_zones_from_scores,
    edge_gain_synthetic,
)
from metrics import edge_l1, edge_corr, mean_abs_change, psnr, ssim
from visualize import save_image, save_zone_map, save_panel_3x3, save_barplot


def _print_guide():
    print('\n=== Synthetic Task ===')
    print('Goal: structural feature extraction under spatially mixed corruption.')
    print('Top-row panels show Raw image plus Global/PDK edge-gain maps relative to Raw.')
    print('PDK rule: preserve already-kept edges, recover edges weakened by global, suppress non-edge regions.')
    print(f"Global baseline kernel: {format_kernel_spec(GLOBAL_KERNEL_SPECS['synthetic'])}")
    print(f"PDK zone kernels     : {format_zone_kernel_specs('synthetic')}")
    print('======================\n')


def run():
    _print_guide()
    save_root = os.path.join(SAVE_ROOT, 'synthetic')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    plot_dir = os.path.join(save_root, 'plots')
    ensure_dir(img_dir); ensure_dir(panel_dir); ensure_dir(plot_dir)
    files = list_image_files(SYNTHETIC['root'], FILE_EXTENSIONS, SYNTHETIC['max_images'])
    if not files:
        raise RuntimeError(f"No images found: {SYNTHETIC['root']}")

    rng = np.random.default_rng(SEED)
    rows = []
    raw_edge_l1s, global_edge_l1s, pdk_edge_l1s = [], [], []
    raw_edge_corrs, global_edge_corrs, pdk_edge_corrs = [], [], []
    raw_psnrs, global_psnrs, pdk_psnrs = [], [], []
    raw_ssims, global_ssims, pdk_ssims = [], [], []

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        clean = load_grayscale(path, SYNTHETIC['image_size'])
        h, w = clean.shape
        field = random_field_map(h, w, rng, SYNTHETIC['corruption_field_sigma'])
        smooth_low = gaussian_filter(clean, sigma=SYNTHETIC['blur_sigma_min'], mode='reflect')
        smooth_high = gaussian_filter(clean, sigma=SYNTHETIC['blur_sigma_max'], mode='reflect')
        corrupted = (1.0 - field) * smooth_low + field * smooth_high
        noise_sigma_map = SYNTHETIC['noise_sigma_min'] + (1.0 - field) * (SYNTHETIC['noise_sigma_max'] - SYNTHETIC['noise_sigma_min'])
        noise = rng.normal(0.0, 1.0, size=clean.shape).astype(np.float32) * noise_sigma_map.astype(np.float32)
        raw = clip01(corrupted + noise)

        global_out = apply_global_kernel_np(raw, 'synthetic')
        wgt = SYNTHETIC['score_weights']
        preserve, recover, suppress = synthetic_scores_np(raw, global_out, wgt['missed'], wgt['coherence'])
        zone = assign_zones_from_scores(preserve, recover, suppress)
        pdk_out = apply_zonewise_pdk_np(raw, zone, 'synthetic')

        global_panel = edge_gain_synthetic(raw, global_out)
        pdk_panel = edge_gain_synthetic(raw, pdk_out)

        raw_e_l1 = edge_l1(raw, clean)
        global_e_l1 = edge_l1(global_out, clean)
        pdk_e_l1 = edge_l1(pdk_out, clean)
        raw_e_corr = edge_corr(raw, clean)
        global_e_corr = edge_corr(global_out, clean)
        pdk_e_corr = edge_corr(pdk_out, clean)
        # PSNR/SSIM: pixel-level & structural fidelity vs clean (GT)
        raw_psnr = psnr(raw, clean)
        global_psnr = psnr(global_out, clean)
        pdk_psnr = psnr(pdk_out, clean)
        raw_ssim = ssim(raw, clean)
        global_ssim = ssim(global_out, clean)
        pdk_ssim = ssim(pdk_out, clean)

        rows.append({
            'name': name,
            'psnr_raw': raw_psnr,
            'psnr_global': global_psnr,
            'psnr_pdk': pdk_psnr,
            'ssim_raw': raw_ssim,
            'ssim_global': global_ssim,
            'ssim_pdk': pdk_ssim,
            'edge_l1_raw': raw_e_l1,
            'edge_l1_global': global_e_l1,
            'edge_l1_pdk': pdk_e_l1,
            'edge_corr_raw': raw_e_corr,
            'edge_corr_global': global_e_corr,
            'edge_corr_pdk': pdk_e_corr,
            'mean_abs_change_global': mean_abs_change(global_out, raw),
            'mean_abs_change_pdk': mean_abs_change(pdk_out, raw),
        })
        raw_edge_l1s.append(raw_e_l1); global_edge_l1s.append(global_e_l1); pdk_edge_l1s.append(pdk_e_l1)
        raw_edge_corrs.append(raw_e_corr); global_edge_corrs.append(global_e_corr); pdk_edge_corrs.append(pdk_e_corr)
        raw_psnrs.append(raw_psnr); global_psnrs.append(global_psnr); pdk_psnrs.append(pdk_psnr)
        raw_ssims.append(raw_ssim); global_ssims.append(global_ssim); pdk_ssims.append(pdk_ssim)

        save_image(os.path.join(img_dir, f'{name}_clean.png'), clean)
        save_image(os.path.join(img_dir, f'{name}_raw.png'), raw)
        save_image(os.path.join(img_dir, f'{name}_global_response.png'), global_panel)
        save_image(os.path.join(img_dir, f'{name}_pdk_response.png'), pdk_panel)
        save_image(os.path.join(img_dir, f'{name}_global_processed.png'), global_out)
        save_image(os.path.join(img_dir, f'{name}_pdk_processed.png'), pdk_out)
        save_image(os.path.join(img_dir, f'{name}_recover_score.png'), recover, cmap='magma')
        save_zone_map(os.path.join(img_dir, f'{name}_zone.png'), zone)
        if idx < SYNTHETIC['panel_limit']:
            save_panel_3x3(
                os.path.join(panel_dir, f'{name}_panel.png'),
                get_global_kernel_matrix('synthetic'),
                recover,
                get_zone_kernel_matrices('synthetic'),
                raw,
                global_panel,
                pdk_panel,
                clean,
                global_out,
                pdk_out,
                reference_title='Clean',
            )
        print(
            f"[{idx+1:02d}] {name} | "
            f"PSNR raw/global/pdk = {raw_psnr:.2f} / {global_psnr:.2f} / {pdk_psnr:.2f} dB | "
            f"SSIM = {raw_ssim:.4f} / {global_ssim:.4f} / {pdk_ssim:.4f} | "
            f"edge_l1 = {raw_e_l1:.4f} / {global_e_l1:.4f} / {pdk_e_l1:.4f}"
        )

    with open(os.path.join(save_root, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    save_barplot(os.path.join(plot_dir, 'psnr.png'), {'Raw': float(np.mean(raw_psnrs)), 'Global': float(np.mean(global_psnrs)), 'PDK': float(np.mean(pdk_psnrs))}, 'PSNR dB (higher is better)')
    save_barplot(os.path.join(plot_dir, 'ssim.png'), {'Raw': float(np.mean(raw_ssims)), 'Global': float(np.mean(global_ssims)), 'PDK': float(np.mean(pdk_ssims))}, 'SSIM (higher is better)')
    save_barplot(os.path.join(plot_dir, 'edge_l1.png'), {'Raw': float(np.mean(raw_edge_l1s)), 'Global': float(np.mean(global_edge_l1s)), 'PDK': float(np.mean(pdk_edge_l1s))}, 'Edge L1 (lower is better)')
    save_barplot(os.path.join(plot_dir, 'edge_corr.png'), {'Raw': float(np.mean(raw_edge_corrs)), 'Global': float(np.mean(global_edge_corrs)), 'PDK': float(np.mean(pdk_edge_corrs))}, 'Edge Corr (higher is better)')

    print(f'\n=== Synthetic Summary ===')
    print(f"PSNR   raw/global/pdk : {np.mean(raw_psnrs):.2f} / {np.mean(global_psnrs):.2f} / {np.mean(pdk_psnrs):.2f} dB")
    print(f"SSIM   raw/global/pdk : {np.mean(raw_ssims):.4f} / {np.mean(global_ssims):.4f} / {np.mean(pdk_ssims):.4f}")
    print(f"EdgeL1 raw/global/pdk : {np.mean(raw_edge_l1s):.4f} / {np.mean(global_edge_l1s):.4f} / {np.mean(pdk_edge_l1s):.4f}")
