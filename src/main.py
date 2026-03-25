import os
import csv
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import CFG
from utils import set_seed
from dataset import list_image_files, BSDSAttentionDataset
from filters import (
    get_zone_kernel_bank,
    apply_kernel_bank_soft,
    apply_kernel_bank_hard,
    apply_global_gaussian,
    gaussian_kernel_np,
)
from fovea_net import FoveaHeatmapNet, params_to_soft_zones, params_to_hard_zones
from losses import total_train_loss, reconstruction_loss, selection_score_from_losses
from metrics import compute_metrics_np
from visualize import save_image, save_zone_map, save_fovea_panel, save_barplot


def device_string(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy().astype(np.float32)


def normalize_map_for_vis(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (x.max() + 1e-8)


def build_loaders():
    train_paths = list_image_files(CFG.TRAIN_ROOT, max_images=CFG.MAX_TRAIN_IMAGES)
    test_paths = list_image_files(CFG.TEST_ROOT, max_images=CFG.MAX_TEST_IMAGES)

    if len(train_paths) < 2:
        raise RuntimeError("Need at least 2 training images.")

    tr_paths, val_paths = train_test_split(
        train_paths,
        test_size=CFG.VAL_RATIO,
        random_state=CFG.SEED,
        shuffle=True,
    )

    train_ds = BSDSAttentionDataset(tr_paths, CFG.IMAGE_SIZE, base_seed=CFG.SEED * 10 + 1)
    val_ds = BSDSAttentionDataset(val_paths, CFG.IMAGE_SIZE, base_seed=CFG.SEED * 10 + 2)
    test_ds = BSDSAttentionDataset(test_paths, CFG.IMAGE_SIZE, base_seed=CFG.SEED * 10 + 3)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def select_best_global_sigma(val_loader, device):
    rows = []
    for sigma in CFG.GLOBAL_SIGMA_CANDIDATES:
        l1_list = []
        grad_list = []
        for batch in val_loader:
            observed = batch["observed"].to(device)
            clean = batch["clean"].to(device)

            pred = apply_global_gaussian(observed, sigma=sigma)
            rec = reconstruction_loss(pred, clean)

            l1_list.append(float(rec["l1"].detach().cpu()))
            grad_list.append(float(rec["grad"].detach().cpu()))

        mean_l1 = float(np.mean(l1_list))
        mean_grad = float(np.mean(grad_list))
        score = selection_score_from_losses(mean_l1, mean_grad)

        rows.append({
            "sigma": float(sigma),
            "l1": mean_l1,
            "grad": mean_grad,
            "score": score,
        })

    best_row = min(rows, key=lambda x: x["score"])
    return best_row, rows


def train_one_epoch(model, loader, optimizer, kernel_bank, device):
    model.train()
    logs = []

    cx_list, cy_list, r1_list, r2_list = [], [], [], []

    for batch in loader:
        observed = batch["observed"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()

        params = model(observed)
        zone_probs, _ = params_to_soft_zones(params, observed.shape[-2], observed.shape[-1], device)
        pred = apply_kernel_bank_soft(observed, zone_probs, kernel_bank)

        loss, log = total_train_loss(pred, clean, zone_probs, params["heatmap_probs"])
        loss.backward()
        optimizer.step()

        logs.append(log)

        cx_list.extend(params["cx"].detach().cpu().numpy().tolist())
        cy_list.extend(params["cy"].detach().cpu().numpy().tolist())
        r1_list.extend(params["r1"].detach().cpu().numpy().tolist())
        r2_list.extend(params["r2"].detach().cpu().numpy().tolist())

    mean_log = {k: float(np.mean([x[k] for x in logs])) for k in logs[0].keys()}
    mean_log["cx"] = float(np.mean(cx_list))
    mean_log["cy"] = float(np.mean(cy_list))
    mean_log["r1"] = float(np.mean(r1_list))
    mean_log["r2"] = float(np.mean(r2_list))
    mean_log["cx_std"] = float(np.std(cx_list))
    mean_log["cy_std"] = float(np.std(cy_list))
    mean_log["r1_std"] = float(np.std(r1_list))
    mean_log["r2_std"] = float(np.std(r2_list))
    return mean_log


@torch.no_grad()
def validate(model, loader, kernel_bank, device):
    model.eval()

    total_l1 = []
    total_grad = []

    cx_list, cy_list, r1_list, r2_list = [], [], [], []

    for batch in loader:
        observed = batch["observed"].to(device)
        clean = batch["clean"].to(device)

        params = model(observed)
        zone_map, _ = params_to_hard_zones(params, observed.shape[-2], observed.shape[-1], device)
        pred = apply_kernel_bank_hard(observed, zone_map, kernel_bank)

        rec = reconstruction_loss(pred, clean)
        total_l1.append(float(rec["l1"].detach().cpu()))
        total_grad.append(float(rec["grad"].detach().cpu()))

        cx_list.extend(params["cx"].detach().cpu().numpy().tolist())
        cy_list.extend(params["cy"].detach().cpu().numpy().tolist())
        r1_list.extend(params["r1"].detach().cpu().numpy().tolist())
        r2_list.extend(params["r2"].detach().cpu().numpy().tolist())

    mean_l1 = float(np.mean(total_l1))
    mean_grad = float(np.mean(total_grad))
    score = selection_score_from_losses(mean_l1, mean_grad)

    return {
        "l1": mean_l1,
        "grad": mean_grad,
        "score": score,
        "cx": float(np.mean(cx_list)),
        "cy": float(np.mean(cy_list)),
        "r1": float(np.mean(r1_list)),
        "r2": float(np.mean(r2_list)),
        "cx_std": float(np.std(cx_list)),
        "cy_std": float(np.std(cy_list)),
        "r1_std": float(np.std(r1_list)),
        "r2_std": float(np.std(r2_list)),
    }


@torch.no_grad()
def evaluate_test(model, test_loader, kernel_bank, global_sigma, device):
    model.eval()

    img_dir = os.path.join(CFG.SAVE_ROOT, "images")
    panel_dir = os.path.join(CFG.SAVE_ROOT, "panels")
    plot_dir = os.path.join(CFG.SAVE_ROOT, "plots")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    rows = []

    psnr_obs = []
    psnr_global = []
    psnr_pdk = []

    ssim_obs = []
    ssim_global = []
    ssim_pdk = []

    grad_obs = []
    grad_global = []
    grad_pdk = []

    for idx, batch in enumerate(test_loader):
        name = batch["name"][0]
        observed = batch["observed"].to(device)
        clean = batch["clean"].to(device)

        params = model(observed)
        zone_map, _ = params_to_hard_zones(params, observed.shape[-2], observed.shape[-1], device)

        out_global = apply_global_gaussian(observed, sigma=global_sigma)
        out_pdk = apply_kernel_bank_hard(observed, zone_map, kernel_bank)

        clean_np = to_numpy_img(clean)
        observed_np = to_numpy_img(observed)
        global_np = to_numpy_img(out_global)
        pdk_np = to_numpy_img(out_pdk)
        zone_map_np = zone_map.detach().cpu().squeeze().numpy().astype(np.int32)
        heatmap_np = normalize_map_for_vis(to_numpy_img(params["heatmap_probs"]))

        m_obs = compute_metrics_np(observed_np, clean_np)
        m_global = compute_metrics_np(global_np, clean_np)
        m_pdk = compute_metrics_np(pdk_np, clean_np)

        rows.append({
            "name": name,
            "cx": float(params["cx"].detach().cpu().item()),
            "cy": float(params["cy"].detach().cpu().item()),
            "r1": float(params["r1"].detach().cpu().item()),
            "r2": float(params["r2"].detach().cpu().item()),
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
            save_image(os.path.join(img_dir, f"{name}_clean.png"), clean_np)
            save_image(os.path.join(img_dir, f"{name}_observed.png"), observed_np)
            save_image(os.path.join(img_dir, f"{name}_fovea_score.png"), heatmap_np, cmap="magma")
            save_zone_map(os.path.join(img_dir, f"{name}_zones.png"), zone_map_np)
            save_image(os.path.join(img_dir, f"{name}_global.png"), global_np)
            save_image(os.path.join(img_dir, f"{name}_pdk.png"), pdk_np)

        if CFG.SAVE_PANELS and idx < CFG.SAVE_MAX_PANELS:
            save_fovea_panel(
                os.path.join(panel_dir, f"{name}_panel.png"),
                clean=clean_np,
                observed=observed_np,
                global_out=global_np,
                pdk_out=pdk_np,
                fovea_map=heatmap_np,
                zone_map=zone_map_np,
            )

        print(
            f"[{idx+1:02d}] {name} | "
            f"cx={params['cx'].item():.3f}, cy={params['cy'].item():.3f}, "
            f"r1={params['r1'].item():.3f}, r2={params['r2'].item():.3f} | "
            f"PSNR obs/global/pdk = {m_obs['psnr']:.3f} / {m_global['psnr']:.3f} / {m_pdk['psnr']:.3f}"
        )

    csv_path = os.path.join(CFG.SAVE_ROOT, "metrics_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_psnr = {
        "Observed": float(np.mean(psnr_obs)),
        "Best Global": float(np.mean(psnr_global)),
        "NN-Foveated PDK": float(np.mean(psnr_pdk)),
    }
    summary_ssim = {
        "Observed": float(np.mean(ssim_obs)),
        "Best Global": float(np.mean(ssim_global)),
        "NN-Foveated PDK": float(np.mean(ssim_pdk)),
    }
    summary_grad = {
        "Observed": float(np.mean(grad_obs)),
        "Best Global": float(np.mean(grad_global)),
        "NN-Foveated PDK": float(np.mean(grad_pdk)),
    }

    save_barplot(os.path.join(plot_dir, "psnr_summary.png"), summary_psnr, ylabel="PSNR (dB)")
    save_barplot(os.path.join(plot_dir, "ssim_summary.png"), summary_ssim, ylabel="SSIM")
    save_barplot(os.path.join(plot_dir, "gradient_summary.png"), summary_grad, ylabel="Gradient L1")

    return summary_psnr, summary_ssim, summary_grad


def main():
    set_seed(CFG.SEED)
    device = device_string(CFG.DEVICE)

    train_loader, val_loader, test_loader = build_loaders()
    kernel_bank = get_zone_kernel_bank(device)

    best_global_row, all_global_rows = select_best_global_sigma(val_loader, device)
    best_global_sigma = best_global_row["sigma"]
    best_global_kernel = gaussian_kernel_np(CFG.KERNEL_SIZE, best_global_sigma)

    print("\n=== Validation Global Search ===")
    for row in all_global_rows:
        print(
            f"sigma={row['sigma']:.2f} | "
            f"L1={row['l1']:.6f} | Grad={row['grad']:.6f} | Score={row['score']:.6f}"
        )

    print("\nSelected best global sigma:")
    print(f"  sigma = {best_global_sigma:.3f}")
    print(f"  kernel size = {CFG.KERNEL_SIZE}x{CFG.KERNEL_SIZE}")
    print("  kernel =")
    print(np.array2string(best_global_kernel, precision=4, suppress_small=True))

    model = FoveaHeatmapNet(in_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    best_state = None
    best_val_score = float("inf")
    ckpt_path = os.path.join(CFG.SAVE_ROOT, "best_fovea_heatmap_model.pt")

    for epoch in range(1, CFG.EPOCHS + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, kernel_bank, device)
        val_log = validate(model, val_loader, kernel_bank, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train total={train_log['total']:.5f}, l1={train_log['l1']:.5f}, "
            f"grad={train_log['grad']:.5f}, ratio={train_log['ratio']:.5f}, entropy={train_log['entropy']:.5f} | "
            f"Train cx={train_log['cx']:.3f}±{train_log['cx_std']:.3f}, "
            f"cy={train_log['cy']:.3f}±{train_log['cy_std']:.3f}, "
            f"r1={train_log['r1']:.3f}±{train_log['r1_std']:.3f}, "
            f"r2={train_log['r2']:.3f}±{train_log['r2_std']:.3f} | "
            f"Val l1={val_log['l1']:.5f}, grad={val_log['grad']:.5f}, score={val_log['score']:.5f} | "
            f"Val cx={val_log['cx']:.3f}±{val_log['cx_std']:.3f}, "
            f"cy={val_log['cy']:.3f}±{val_log['cy_std']:.3f}, "
            f"r1={val_log['r1']:.3f}±{val_log['r1_std']:.3f}, "
            f"r2={val_log['r2']:.3f}±{val_log['r2_std']:.3f}"
        )

        if val_log["score"] < best_val_score:
            best_val_score = val_log["score"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "val_score": best_val_score,
                    "best_global_sigma": best_global_sigma,
                },
                ckpt_path,
            )

    if best_state is None:
        raise RuntimeError("Training failed: no checkpoint saved.")

    model.load_state_dict(best_state)

    summary_psnr, summary_ssim, summary_grad = evaluate_test(
        model=model,
        test_loader=test_loader,
        kernel_bank=kernel_bank,
        global_sigma=best_global_sigma,
        device=device,
    )

    print("\n=== Final Test Summary ===")
    print("PSNR:", summary_psnr)
    print("SSIM:", summary_ssim)
    print("Gradient:", summary_grad)
    print(f"Saved to: {CFG.SAVE_ROOT}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()