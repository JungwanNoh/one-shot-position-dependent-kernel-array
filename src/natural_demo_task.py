import os
from config import SAVE_ROOT, FILE_EXTENSIONS, GLOBAL_SIGMA, NATURAL, ZONE_KERNEL_SPECS
from utils import list_image_files, load_grayscale, ensure_dir
from kernels import apply_global_gaussian_np, apply_zonewise_pdk_np
from zone_generators import global_first_score_np, clustered_zone_from_score_np
from visualize import save_image, save_zone_map, save_panel


def run():
    save_root = os.path.join(SAVE_ROOT, 'natural_demo')
    img_dir = os.path.join(save_root, 'images')
    panel_dir = os.path.join(save_root, 'panels')
    ensure_dir(img_dir); ensure_dir(panel_dir)
    files = list_image_files(NATURAL['root'], FILE_EXTENSIONS, NATURAL['max_images'])
    if not files:
        raise RuntimeError(f"No images found: {NATURAL['root']}")
    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_grayscale(path, NATURAL['image_size'])
        global_out = apply_global_gaussian_np(img, GLOBAL_SIGMA['natural'])
        w = NATURAL['score_weights']
        score = global_first_score_np(img, global_out, w['residual'], w['gradient'], None, 0.0)
        zone = clustered_zone_from_score_np(score)
        pdk_out = apply_zonewise_pdk_np(img, zone, ZONE_KERNEL_SPECS)
        save_image(os.path.join(img_dir, f'{name}_score.png'), score, cmap='magma')
        save_zone_map(os.path.join(img_dir, f'{name}_zone.png'), zone)
        save_image(os.path.join(img_dir, f'{name}_global.png'), global_out)
        save_image(os.path.join(img_dir, f'{name}_pdk.png'), pdk_out)
        if idx < 8:
            save_panel(os.path.join(panel_dir, f'{name}_panel.png'), global_out, pdk_out, zone, gt=None, input_img=img)
        print(f"[{idx+1:02d}] {name} done")
