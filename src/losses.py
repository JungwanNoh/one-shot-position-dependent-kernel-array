import torch
import torch.nn.functional as F

from config import CFG


def sobel_kernels(device: str):
    kx = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=torch.float32, device=device
    )[None, None, :, :]
    ky = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=torch.float32, device=device
    )[None, None, :, :]
    return kx, ky


def gradient_mag(x: torch.Tensor) -> torch.Tensor:
    kx, ky = sobel_kernels(x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gp = gradient_mag(pred)
    gt = gradient_mag(target)
    return F.l1_loss(gp, gt)


def zone_ratio_loss(zone_probs: torch.Tensor) -> torch.Tensor:
    actual = zone_probs.mean(dim=(0, 2, 3))
    target = torch.tensor(CFG.TARGET_ZONE_RATIOS, dtype=zone_probs.dtype, device=zone_probs.device)
    return F.l1_loss(actual, target)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    l1 = F.l1_loss(pred, target)
    grad = gradient_l1_loss(pred, target)
    return {"l1": l1, "grad": grad}


def total_train_loss(pred: torch.Tensor, target: torch.Tensor, zone_probs: torch.Tensor):
    rec = reconstruction_loss(pred, target)
    ratio = zone_ratio_loss(zone_probs)

    total = (
        CFG.LOSS_W_L1 * rec["l1"]
        + CFG.LOSS_W_GRAD * rec["grad"]
        + CFG.LOSS_W_RATIO * ratio
    )

    log = {
        "total": float(total.detach().cpu()),
        "l1": float(rec["l1"].detach().cpu()),
        "grad": float(rec["grad"].detach().cpu()),
        "ratio": float(ratio.detach().cpu()),
    }
    return total, log


def selection_score_from_losses(l1_value: float, grad_value: float) -> float:
    return CFG.SELECT_W_L1 * l1_value + CFG.SELECT_W_GRAD * grad_value