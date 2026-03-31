import os
import csv
import numpy as np

from config import SAVE_ROOT, FILE_EXTENSIONS, NATURAL, GLOBAL_KERNEL_SPECS
from utils import list_image_files, load_grayscale, ensure_dir
from kernels import (
    apply_global_kernel_np,
    apply_zonewise_pdk_np,
    format_kernel_spec,
    format_zone_kernel_specs,
    get_global_kernel_matrix,
    get_zone_kernel_matrices,
)
from zone_generators import natural_scores_np, assign_zones_from_scores, edge_response_natural
from metrics import mean_abs_change
from visualize import save_image, save_zone_map, save_panel_3x3, save_barplot


def _print_guide():
    print('\n=== Natural Task ===')
    print('Goal: make object-relevant contours more visible for recognition-oriented preprocessing.')
    print('Top-row panels show Raw image plus Global/PDK edge-response maps extracted from each processed image.')
    print('PDK rule: preserve stable contours, recover global-missed contour candidates, suppress clutter/background refinement.')
    print(f"Global baseline kernel: {format_kernel_spec(GLOBAL_KERNEL_SPECS['natural'])}")
    print(f"PDK zone kernels     : {format_zone_kernel_specs('natural')}")
    print('====================\n')


def run():
    _print_guide()
    save_root = os.path.join(SAVE_ROOT, 'natural')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    plot_dir = os.path.join(save_root, 'plots')
    ensure_dir(img_dir); ensure_dir(panel_dir); ensure_dir(plot_dir)
    files = list_image_files(NATURAL['root'], FILE_EXTENSIONS, NATURAL['max_images'])
    if not files:
        raise RuntimeError(f"No images found: {NATURAL['root']}")

    rows = []
    mean_global_changes, mean_pdk_changes = [], []
    mean_global_resp, mean_pdk_resp = [], []

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        raw = load_grayscale(path, NATURAL['image_size'])
        global_out = apply_global_kernel_np(raw, 'natural')
        wgt = NATURAL['score_weights']
        preserve, recover, suppress = natural_scores_np(raw, global_out, wgt['missed'], wgt['coherence'], wgt['texture_penalty'])
        zone = assign_zones_from_scores(preserve, recover, suppress)
        pdk_out = apply_zonewise_pdk_np(raw, zone, 'natural')

        global_panel = edge_response_natural(global_out)
        pdk_panel = edge_response_natural(pdk_out)
        mean_global_change = mean_abs_change(global_out, raw)
        mean_pdk_change = mean_abs_change(pdk_out, raw)
        rows.append({
            'name': name,
            'mean_abs_change_global': mean_global_change,
            'mean_abs_change_pdk': mean_pdk_change,
            'mean_response_global': float(global_panel.mean()),
            'mean_response_pdk': float(pdk_panel.mean()),
        })
        mean_global_changes.append(mean_global_change)
        mean_pdk_changes.append(mean_pdk_change)
        mean_global_resp.append(float(global_panel.mean()))
        mean_pdk_resp.append(float(pdk_panel.mean()))

        save_image(os.path.join(img_dir, f'{name}_raw.png'), raw)
        save_image(os.path.join(img_dir, f'{name}_global_response.png'), global_panel)
        save_image(os.path.join(img_dir, f'{name}_pdk_response.png'), pdk_panel)
        save_image(os.path.join(img_dir, f'{name}_global_processed.png'), global_out)
        save_image(os.path.join(img_dir, f'{name}_pdk_processed.png'), pdk_out)
        save_image(os.path.join(img_dir, f'{name}_recover_score.png'), recover, cmap='magma')
        save_zone_map(os.path.join(img_dir, f'{name}_zone.png'), zone)
        if idx < NATURAL['panel_limit']:
            save_panel_3x3(
                os.path.join(panel_dir, f'{name}_panel.png'),
                get_global_kernel_matrix('natural'),
                recover,
                get_zone_kernel_matrices('natural'),
                raw,
                global_panel,
                pdk_panel,
                raw,
                global_out,
                pdk_out,
                reference_title='Raw reference',
            )
        print(f"[{idx+1:02d}] {name} | mean_change global/pdk = {mean_global_change:.4f} / {mean_pdk_change:.4f}")

    with open(os.path.join(save_root, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    save_barplot(os.path.join(plot_dir, 'mean_abs_change.png'), {'Global': float(np.mean(mean_global_changes)), 'PDK': float(np.mean(mean_pdk_changes))}, 'Mean |processed - raw|')
    save_barplot(os.path.join(plot_dir, 'mean_response.png'), {'Global': float(np.mean(mean_global_resp)), 'PDK': float(np.mean(mean_pdk_resp))}, 'Mean edge-map intensity')
