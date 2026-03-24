import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim


def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def l1_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def psnr_np(pred: np.ndarray, target: np.ndarray) -> float:
    mse = mse_np(pred, target)
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ssim(target, pred, data_range=1.0))


def gradient_mag_np(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def gradient_l1_np(pred: np.ndarray, target: np.ndarray) -> float:
    gp = gradient_mag_np(pred)
    gt = gradient_mag_np(target)
    return float(np.mean(np.abs(gp - gt)))


def compute_metrics_np(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "l1": l1_np(pred, target),
        "mse": mse_np(pred, target),
        "psnr": psnr_np(pred, target),
        "ssim": ssim_np(pred, target),
        "grad": gradient_l1_np(pred, target),
    }