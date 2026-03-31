import numpy as np
from scipy.ndimage import sobel, gaussian_filter
from utils import normalize01


def gradient_mag(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode='reflect')
    gy = sobel(img, axis=0, mode='reflect')
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def edge_l1(pred_img: np.ndarray, target_img: np.ndarray) -> float:
    pred = normalize01(gradient_mag(pred_img))
    tgt = normalize01(gradient_mag(target_img))
    return float(np.mean(np.abs(pred - tgt)))


def edge_corr(pred_img: np.ndarray, target_img: np.ndarray) -> float:
    pred = normalize01(gradient_mag(pred_img)).reshape(-1)
    tgt = normalize01(gradient_mag(target_img)).reshape(-1)
    pred = pred - pred.mean()
    tgt = tgt - tgt.mean()
    denom = np.sqrt((pred ** 2).sum() * (tgt ** 2).sum()) + 1e-8
    return float((pred * tgt).sum() / denom)


def mean_abs_change(processed: np.ndarray, raw: np.ndarray) -> float:
    return float(np.mean(np.abs(processed - raw)))


# ------------------------------------------------------------------
# PSNR
# Peak Signal-to-Noise Ratio (dB).
# Measures pixel-level fidelity vs GT. Higher is better.
# Formula: 10 * log10(MAX^2 / MSE), MAX=1 for float [0,1] images.
# Typical range: <20 noticeable degradation, 25-35 acceptable, >35 good.
# ------------------------------------------------------------------
def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse_val = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    if mse_val < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse_val))


# ------------------------------------------------------------------
# SSIM  (Wang et al., 2004)
# Structural Similarity Index. Measures luminance, contrast, structure.
# Range [0, 1]. 1 = identical. Sensitive to structural distortions
# that PSNR misses (e.g. zone-boundary artifacts from PDK).
# Implemented with Gaussian window via scipy (no skimage needed).
# ------------------------------------------------------------------
def ssim(pred: np.ndarray, target: np.ndarray, sigma: float = 1.5) -> float:
    # Stability constants (K1=0.01, K2=0.03, L=1.0 data range)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    mu_x = gaussian_filter(pred, sigma=sigma, mode='reflect')
    mu_y = gaussian_filter(target, sigma=sigma, mode='reflect')

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(pred * pred, sigma=sigma, mode='reflect') - mu_x2
    sigma_y2 = gaussian_filter(target * target, sigma=sigma, mode='reflect') - mu_y2
    sigma_xy = gaussian_filter(pred * target, sigma=sigma, mode='reflect') - mu_xy

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean())


# ------------------------------------------------------------------
# Convenience: compute all GT-reference metrics at once.
# Returns dict with keys: psnr, ssim, edge_l1, edge_corr, mean_abs_change
# ------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, target: np.ndarray, raw: np.ndarray = None) -> dict:
    result = {
        'psnr': psnr(pred, target),
        'ssim': ssim(pred, target),
        'edge_l1': edge_l1(pred, target),
        'edge_corr': edge_corr(pred, target),
    }
    if raw is not None:
        result['mean_abs_change'] = mean_abs_change(pred, raw)
    return result
