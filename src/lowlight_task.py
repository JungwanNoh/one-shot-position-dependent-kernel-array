import os
import csv
import numpy as np

from config import SAVE_ROOT, FILE_EXTENSIONS, LOWLIGHT, GLOBAL_KERNEL_SPECS
from utils import list_image_files, load_grayscale, ensure_dir
from kernels import (
    apply_global_kernel_np,
    apply_zonewise_pdk_np,
    format_kernel_spec,
    format_zone_kernel_specs,
    get_global_kernel_matrix,
    get_zone_kernel_matrices,
)
from zone_generators import (
    lowlight_scores_np,
    assign_zones_from_scores,
    edge_gain_lowlight,
)
from metrics import edge_l1, edge_corr, mean_abs_change, psnr, ssim
from visualize import save_image, save_zone_map, save_panel_3x3, save_barplot


def _print_guide():
    print('\n=== Lowlight Task ===')
    print('Goal: make object-relevant low-light contours more visible while suppressing noisy background regions.')
    print('Top-row panels show Raw image plus Global/PDK edge-gain maps relative to Raw.')
    print('PDK rule: preserve already-visible contours, recover global-missed edge candidates, suppress dark noisy regions.')
    print(f"Global baseline kernel: {format_kernel_spec(GLOBAL_KERNEL_SPECS['lowlight'])}")
    print(f"PDK zone kernels     : {format_zone_kernel_specs('lowlight')}")
    print('====================\n')


def run():
    _print_guide()
    save_root = os.path.join(SAVE_ROOT, 'lowlight')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    plot_dir = os.path.join(save_root, 'plots')
    ensure_dir(img_dir); ensure_dir(panel_dir); ensure_dir(plot_dir)
    low_files = list_image_files(LOWLIGHT['input_root'], FILE_EXTENSIONS, LOWLIGHT['max_images'])
    gt_files = list_image_files(LOWLIGHT['gt_root'], FILE_EXTENSIONS, LOWLIGHT['max_images'])
    gt_map = {os.path.basename(p): p for p in gt_files}
    pairs = []
    for lp in low_files:
        name = os.path.basename(lp)
        if name in gt_map:
            pairs.append((lp, gt_map[name]))
    if not pairs:
        raise RuntimeError(f'No low/gt pairs found under {LOWLIGHT["input_root"]} and {LOWLIGHT["gt_root"]}')

    rows = []
    raw_edge_l1s, global_edge_l1s, pdk_edge_l1s = [], [], []
    raw_edge_corrs, global_edge_corrs, pdk_edge_corrs = [], [], []
    raw_psnrs, global_psnrs, pdk_psnrs = [], [], []
    raw_ssims, global_ssims, pdk_ssims = [], [], []

    for idx, (low_path, gt_path) in enumerate(pairs):
        name = os.path.splitext(os.path.basename(low_path))[0]
        raw = load_grayscale(low_path, LOWLIGHT['image_size'])
        gt = load_grayscale(gt_path, LOWLIGHT['image_size'])
        global_out = apply_global_kernel_np(raw, 'lowlight')
        wgt = LOWLIGHT['score_weights']
        preserve, recover, suppress = lowlight_scores_np(raw, global_out, wgt['missed'], wgt['coherence'], wgt['brightness'], wgt['noise'])
        zone = assign_zones_from_scores(preserve, recover, suppress)
        pdk_out = apply_zonewise_pdk_np(raw, zone, 'lowlight')

        global_panel = edge_gain_lowlight(raw, global_out)
        pdk_panel = edge_gain_lowlight(raw, pdk_out)

        raw_e_l1 = edge_l1(raw, gt)
        global_e_l1 = edge_l1(global_out, gt)
        pdk_e_l1 = edge_l1(pdk_out, gt)
        raw_e_corr = edge_corr(raw, gt)
        global_e_corr = edge_corr(global_out, gt)
        pdk_e_corr = edge_corr(pdk_out, gt)
        # PSNR/SSIM: pixel-level & structural fidelity vs GT (high-light reference)
        raw_p = psnr(raw, gt)
        global_p = psnr(global_out, gt)
        pdk_p = psnr(pdk_out, gt)
        raw_s = ssim(raw, gt)
        global_s = ssim(global_out, gt)
        pdk_s = ssim(pdk_out, gt)
        rows.append({
            'name': name,
            'psnr_raw': raw_p,
            'psnr_global': global_p,
            'psnr_pdk': pdk_p,
            'ssim_raw': raw_s,
            'ssim_global': global_s,
            'ssim_pdk': pdk_s,
            'edge_l1_raw': raw_e_l1,
            'edge_l1_global': global_e_l1,
            'edge_l1_pdk': pdk_e_l1,
            'edge_corr_raw': raw_e_corr,
            'edge_corr_global': global_e_corr,
            'edge_corr_pdk': pdk_e_corr,
            'mean_input': float(raw.mean()),
            'mean_gt': float(gt.mean()),
            'mean_global': float(global_out.mean()),
            'mean_pdk': float(pdk_out.mean()),
            'mean_abs_change_global': mean_abs_change(global_out, raw),
            'mean_abs_change_pdk': mean_abs_change(pdk_out, raw),
        })
        raw_edge_l1s.append(raw_e_l1); global_edge_l1s.append(global_e_l1); pdk_edge_l1s.append(pdk_e_l1)
        raw_edge_corrs.append(raw_e_corr); global_edge_corrs.append(global_e_corr); pdk_edge_corrs.append(pdk_e_corr)
        raw_psnrs.append(raw_p); global_psnrs.append(global_p); pdk_psnrs.append(pdk_p)
        raw_ssims.append(raw_s); global_ssims.append(global_s); pdk_ssims.append(pdk_s)

        save_image(os.path.join(img_dir, f'{name}_gt.png'), gt)
        save_image(os.path.join(img_dir, f'{name}_raw.png'), raw)
        save_image(os.path.join(img_dir, f'{name}_global_response.png'), global_panel)
        save_image(os.path.join(img_dir, f'{name}_pdk_response.png'), pdk_panel)
        save_image(os.path.join(img_dir, f'{name}_global_processed.png'), global_out)
        save_image(os.path.join(img_dir, f'{name}_pdk_processed.png'), pdk_out)
        save_image(os.path.join(img_dir, f'{name}_recover_score.png'), recover, cmap='magma')
        save_zone_map(os.path.join(img_dir, f'{name}_zone.png'), zone)
        if idx < LOWLIGHT['panel_limit']:
            save_panel_3x3(
                os.path.join(panel_dir, f'{name}_panel.png'),
                get_global_kernel_matrix('lowlight'),
                recover,
                get_zone_kernel_matrices('lowlight'),
                raw,
                global_panel,
                pdk_panel,
                gt,
                global_out,
                pdk_out,
                reference_title='GT',
            )
        print(
            f"[{idx+1:02d}] {name} | "
            f"PSNR raw/global/pdk = {raw_p:.2f} / {global_p:.2f} / {pdk_p:.2f} dB | "
            f"SSIM = {raw_s:.4f} / {global_s:.4f} / {pdk_s:.4f} | "
            f"edge_l1 = {raw_e_l1:.4f} / {global_e_l1:.4f} / {pdk_e_l1:.4f}"
        )

    with open(os.path.join(save_root, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    save_barplot(os.path.join(plot_dir, 'psnr.png'), {'Raw': float(np.mean(raw_psnrs)), 'Global': float(np.mean(global_psnrs)), 'PDK': float(np.mean(pdk_psnrs))}, 'PSNR dB (higher is better)')
    save_barplot(os.path.join(plot_dir, 'ssim.png'), {'Raw': float(np.mean(raw_ssims)), 'Global': float(np.mean(global_ssims)), 'PDK': float(np.mean(pdk_ssims))}, 'SSIM (higher is better)')
    save_barplot(os.path.join(plot_dir, 'edge_l1.png'), {'Raw': float(np.mean(raw_edge_l1s)), 'Global': float(np.mean(global_edge_l1s)), 'PDK': float(np.mean(pdk_edge_l1s))}, 'Edge L1 (lower is better)')
    save_barplot(os.path.join(plot_dir, 'edge_corr.png'), {'Raw': float(np.mean(raw_edge_corrs)), 'Global': float(np.mean(global_edge_corrs)), 'PDK': float(np.mean(pdk_edge_corrs))}, 'Edge Corr (higher is better)')

    print(f'\n=== Lowlight Summary ===')
    print(f"PSNR   raw/global/pdk : {np.mean(raw_psnrs):.2f} / {np.mean(global_psnrs):.2f} / {np.mean(pdk_psnrs):.2f} dB")
    print(f"SSIM   raw/global/pdk : {np.mean(raw_ssims):.4f} / {np.mean(global_ssims):.4f} / {np.mean(pdk_ssims):.4f}")
    print(f"EdgeL1 raw/global/pdk : {np.mean(raw_edge_l1s):.4f} / {np.mean(global_edge_l1s):.4f} / {np.mean(pdk_edge_l1s):.4f}")
