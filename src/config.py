import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


@dataclass
class Config:
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    VAL_ROOT: str = "../dat/BSDS300/images/train"
    TEST_ROOT: str = "../dat/BSDS300/images/test"
    SAVE_ROOT: str = "../res/gaze_sensor_assisted_zoning"

    FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    MAX_VAL_IMAGES: int | None = 50
    MAX_TEST_IMAGES: int | None = 30

    # Optional gaze input CSV
    # format: name,x_norm,y_norm
    GAZE_CSV_PATH: str = "../dat/gaze_points.csv"
    DEFAULT_GAZE: Tuple[float, float] = (0.5, 0.5)

    SEED: int = 42

    # ---------------------------------------------------------
    # Synthetic observed image generation
    # ---------------------------------------------------------
    MAP_MODE: str = "radial"   # "radial" or "blocky"
    RADIAL_CENTER: Tuple[float, float] = (0.5, 0.5)
    RADIAL_FALLOFF: float = 1.6
    V_MAX: float = 1.0
    V_MIN: float = 0.45

    NOISE_SIGMA_MIN: float = 0.01
    NOISE_SIGMA_ALPHA: float = 0.08

    # ---------------------------------------------------------
    # Gaze -> base zone
    # zone id: 0=attention, 1=intermediate, 2=around
    # ---------------------------------------------------------
    GAZE_SIGMA: float = 0.12
    GAZE_ATTN_TH: float = 0.55
    GAZE_INTER_TH: float = 0.20

    # ---------------------------------------------------------
    # Sensor-assisted demotion
    # If reliability is low, demote zone priority
    # ---------------------------------------------------------
    SENSOR_DEMOTE_TH: float = 0.62
    SENSOR_FORCE_AROUND_TH: float = 0.50

    # ---------------------------------------------------------
    # Kernel setup
    # ---------------------------------------------------------
    KERNEL_SIZE: int = 3
    GLOBAL_SIGMA_CANDIDATES: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8)

    PDK_PRESETS: List[Dict] = field(
        default_factory=lambda: [
            {
                "name": "preset_v1",
                "zone_specs": {
                    0: {"family": "unsharp", "sigma": 0.9, "amount": 0.45},  # attention
                    1: {"family": "binomial"},                                # intermediate
                    2: {"family": "gaussian", "sigma": 1.6},                  # around
                }
            },
            {
                "name": "preset_v2",
                "zone_specs": {
                    0: {"family": "unsharp", "sigma": 0.8, "amount": 0.35},
                    1: {"family": "gaussian", "sigma": 0.9},
                    2: {"family": "gaussian", "sigma": 1.8},
                }
            },
            {
                "name": "preset_v3",
                "zone_specs": {
                    0: {"family": "binomial"},
                    1: {"family": "binomial"},
                    2: {"family": "gaussian", "sigma": 1.5},
                }
            },
        ]
    )

    # ---------------------------------------------------------
    # Selection criterion
    # ---------------------------------------------------------
    SELECT_W_L1: float = 0.70
    SELECT_W_GRAD: float = 0.30

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    SAVE_INDIVIDUAL_IMAGES: bool = True
    SAVE_PANELS: bool = True
    SAVE_MAX_PANELS: int = 8


CFG = Config()
os.makedirs(CFG.SAVE_ROOT, exist_ok=True)