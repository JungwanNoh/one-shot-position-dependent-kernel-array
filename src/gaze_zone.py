import numpy as np
from config import CFG


def generate_gaze_map(h: int, w: int, gaze_xy: tuple[float, float], sigma: float | None = None) -> np.ndarray:
    if sigma is None:
        sigma = CFG.GAZE_SIGMA

    gx, gy = gaze_xy
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
    gaze_map = np.exp(-dist2 / (2.0 * sigma * sigma + 1e-12)).astype(np.float32)
    return gaze_map


def base_zone_from_gaze_map(gaze_map: np.ndarray) -> np.ndarray:
    zone_map = np.full_like(gaze_map, 2, dtype=np.int32)  # around
    zone_map[gaze_map >= CFG.GAZE_INTER_TH] = 1           # intermediate
    zone_map[gaze_map >= CFG.GAZE_ATTN_TH] = 0            # attention
    return zone_map


def apply_sensor_assist(zone_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    """
    If local reliability is low, demote the zone.
    0=attention, 1=intermediate, 2=around
    """
    out = zone_map.copy()

    # moderate demotion
    demote_mask = v_map < CFG.SENSOR_DEMOTE_TH
    out[demote_mask] = np.minimum(out[demote_mask] + 1, 2)

    # force low-reliability pixels to around
    force_around_mask = v_map < CFG.SENSOR_FORCE_AROUND_TH
    out[force_around_mask] = 2

    return out.astype(np.int32)


def build_zone_maps(gaze_xy: tuple[float, float], v_map: np.ndarray):
    h, w = v_map.shape
    gaze_map = generate_gaze_map(h, w, gaze_xy)
    base_zone = base_zone_from_gaze_map(gaze_map)
    final_zone = apply_sensor_assist(base_zone, v_map)
    return gaze_map, base_zone, final_zone