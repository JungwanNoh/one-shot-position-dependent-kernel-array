import os

from config import CFG
from utils import list_image_files, load_grayscale, ensure_dir
from kernels import apply_global_gaussian_np, apply_zonewise_pdk_np
from zone_generators import natural_score, quantize_score_to_zone
from visualize import save_image, save_zone_map, save_panel


def run():
    save_root = os.path.join(CFG.SAVE_ROOT, "natural_demo")
    img_dir = os.path.join(save_root, "images")
    panel_dir = os.path.join(save_root, "panels")
    ensure_dir(img_dir); ensure_dir(panel_dir)

    files = list_image_files(CFG.NATURAL_ROOT, CFG.FILE_EXTENSIONS, CFG.NATURAL_MAX_IMAGES)
    if not files:
        raise RuntimeError(f"No images found: {CFG.NATURAL_ROOT}")

    for idx, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        img = load_grayscale(path, CFG.IMAGE_SIZE)

        global_out = apply_global_gaussian_np(img, CFG.GLOBAL_SIGMA_SYNTHETIC)
        score = natural_score(img, global_out, CFG.NATURAL_SCORE_WEIGHTS)
        zone = quantize_score_to_zone(score)
        pdk_out = apply_zonewise_pdk_np(img, zone, CFG.ZONE_KERNEL_SPECS)

        save_image(os.path.join(img_dir, f"{name}_raw.png"), img)
        save_image(os.path.join(img_dir, f"{name}_score.png"), score, cmap="magma")
        save_zone_map(os.path.join(img_dir, f"{name}_zone.png"), zone)
        save_image(os.path.join(img_dir, f"{name}_global.png"), global_out)
        save_image(os.path.join(img_dir, f"{name}_pdk.png"), pdk_out)

        if idx < CFG.SAVE_MAX_PANELS:
            save_panel(os.path.join(panel_dir, f"{name}_panel.png"), img, global_out, pdk_out, score, zone, None)

        print(f"[{idx+1:02d}] {name} done")