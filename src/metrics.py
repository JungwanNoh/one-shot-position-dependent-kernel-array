import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    m = mse(pred, target)
    if m < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / m))


def ssim_metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(ssim(target, pred, data_range=1.0))


def grad_mag(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode="reflect")
    gy = sobel(img, axis=0, mode="reflect")
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def grad_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(grad_mag(pred) - grad_mag(target))))


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "l1": l1(pred, target),
        "mse": mse(pred, target),
        "psnr": psnr(pred, target),
        "ssim": ssim_metric(pred, target),
        "grad": grad_l1(pred, target),
    }
