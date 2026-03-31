import numpy as np
from scipy.ndimage import sobel, gaussian_filter, label, binary_dilation
from utils import normalize01
from config import ZONE, SYNTHETIC

def gradient_mag_np(img: np.ndarray) -> np.ndarray:
    gx = sobel(img, axis=1, mode='reflect')
    gy = sobel(img, axis=0, mode='reflect')
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)

def synthetic_variability_map(h: int, w: int):
    mode = SYNTHETIC['map_mode']
    vmin = SYNTHETIC['vmin']
    vmax = SYNTHETIC['vmax']
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing='ij',
    )
    if mode == 'radial':
        cx, cy = SYNTHETIC['radial_center']
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rr = rr / (rr.max() + 1e-8)
        fall = rr ** SYNTHETIC['radial_falloff']
        v = vmax - (vmax - vmin) * fall
    else:
        v = np.ones((h, w), dtype=np.float32) * 0.95
        h1, h2 = h // 3, 2 * h // 3
        w1, w2 = w // 3, 2 * w // 3
        v[:h1, :w1] = 0.55
        v[:h1, w2:] = 0.65
        v[h2:, :w1] = 0.60
        v[h2:, w2:] = 0.50
        v[h1:h2, w1:w2] = 0.95
    return v.astype(np.float32)

def global_first_score_np(observed: np.ndarray, global_out: np.ndarray, residual_w: float, gradient_w: float, extra_map: np.ndarray | None = None, extra_w: float = 0.0):
    residual = normalize01(np.abs(observed - global_out))
    grad = normalize01(gradient_mag_np(global_out))
    score = residual_w * residual + gradient_w * grad
    if extra_map is not None and extra_w > 0.0:
        score = score + extra_w * normalize01(extra_map)
    score = gaussian_filter(score.astype(np.float32), sigma=ZONE['score_blur_sigma'])
    return normalize01(score)

def clustered_zone_from_score_np(score: np.ndarray):
    hi = np.percentile(score, ZONE['high_percentile'])
    md = np.percentile(score, ZONE['mid_percentile'])
    strong = score >= hi
    weak = score >= md
    comp_map, num = label(weak)
    attention = np.zeros_like(weak, dtype=bool)
    for cid in range(1, num + 1):
        comp = comp_map == cid
        if comp.sum() < ZONE['min_component_area']:
            continue
        if np.any(strong & comp):
            attention |= comp
    ring = binary_dilation(attention, iterations=ZONE['dilation_iters']) & (~attention)
    intermediate = ring | (weak & (~attention))
    zone = np.full(score.shape, 2, dtype=np.int32)
    zone[intermediate] = 1
    zone[attention] = 0
    return zone

def rgb_to_gray_torch(x):
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

def gradient_mag_torch(x):
    import torch
    import torch.nn.functional as F
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device)[None,None,:,:]
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device)[None,None,:,:]
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def normalize01_torch(x):
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1, keepdim=True).values.view(b,1,1,1)
    mx = flat.max(dim=1, keepdim=True).values.view(b,1,1,1)
    return (x - mn) / (mx - mn + 1e-8)

def classification_score_torch(x, global_x, residual_w: float, gradient_w: float):
    gray_raw = rgb_to_gray_torch(x)
    gray_global = rgb_to_gray_torch(global_x)
    residual = normalize01_torch((gray_raw - gray_global).abs())
    grad = normalize01_torch(gradient_mag_torch(gray_global))
    score = residual_w * residual + gradient_w * grad
    return normalize01_torch(score)

def quantize_score_to_zone_torch(score, high_percentile: float, mid_percentile: float):
    import torch
    b = score.shape[0]
    flat = score.view(b, -1)
    hi = torch.quantile(flat, q=high_percentile / 100.0, dim=1).view(b,1,1,1)
    md = torch.quantile(flat, q=mid_percentile / 100.0, dim=1).view(b,1,1,1)
    zone = torch.full_like(score, 2, dtype=torch.long)
    zone = torch.where(score >= md, torch.ones_like(zone), zone)
    zone = torch.where(score >= hi, torch.zeros_like(zone), zone)
    return zone.squeeze(1)
