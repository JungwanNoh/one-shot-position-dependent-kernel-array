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


def compute_regionwise_mse(pred: np.ndarray, target: np.ndarray, region_map: np.ndarray) -> dict:
    result = {}
    for region_id in [0, 1, 2]:
        mask = (region_map == region_id)
        if mask.sum() == 0:
            result[region_id] = float("nan")
        else:
            result[region_id] = float(np.mean((pred[mask] - target[mask]) ** 2))
    return result


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    mag = np.sqrt(gx**2 + gy**2)
    return mag.astype(np.float32)


def compute_gradient_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_g = gradient_magnitude(pred)
    tgt_g = gradient_magnitude(target)
    return float(np.mean(np.abs(pred_g - tgt_g)))