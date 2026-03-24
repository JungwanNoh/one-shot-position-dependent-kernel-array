import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = compute_mse(pred, target)
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ssim(target, pred, data_range=1.0))


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    mag = np.sqrt(gx**2 + gy**2)
    return mag.astype(np.float32)


def compute_gradient_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_g = gradient_magnitude(pred)
    tgt_g = gradient_magnitude(target)
    return float(np.mean(np.abs(pred_g - tgt_g)))


def compute_edge_mask(target: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    grad = gradient_magnitude(target)
    th = np.percentile(grad, percentile)
    return grad >= th


def compute_edge_mse(pred: np.ndarray, target: np.ndarray, percentile: float = 75.0) -> float:
    edge_mask = compute_edge_mask(target, percentile=percentile)
    if edge_mask.sum() == 0:
        return 0.0
    return float(np.mean((pred[edge_mask] - target[edge_mask]) ** 2))


def compute_regionwise_mse(pred: np.ndarray, target: np.ndarray, region_map: np.ndarray) -> dict:
    result = {}
    for region_id in [0, 1, 2]:
        mask = (region_map == region_id)
        if mask.sum() == 0:
            result[region_id] = float("nan")
        else:
            result[region_id] = float(np.mean((pred[mask] - target[mask]) ** 2))
    return result


def compute_regionwise_mse_mean(pred: np.ndarray, target: np.ndarray, region_map: np.ndarray) -> float:
    reg = compute_regionwise_mse(pred, target, region_map)
    vals = [v for v in reg.values() if not np.isnan(v)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, region_map: np.ndarray, edge_percentile: float = 75.0) -> dict:
    return {
        "mse": compute_mse(pred, target),
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "grad": compute_gradient_error(pred, target),
        "edge_mse": compute_edge_mse(pred, target, percentile=edge_percentile),
        "region_mse": compute_regionwise_mse_mean(pred, target, region_map),
    }