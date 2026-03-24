import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List


@dataclass
class Config:
    # =========================================================
    # Paths
    # =========================================================
    DATA_ROOT: str = "../dat/BSDS300/images/test"
    SAVE_ROOT: str = "../res/variability_simulation_content_aware"

    # =========================================================
    # Dataset / split
    # =========================================================
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    MAX_IMAGES: int = 30
    VAL_RATIO: float = 0.30
    FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    # =========================================================
    # Reproducibility
    # =========================================================
    SEED: int = 42

    # =========================================================
    # Variability map settings
    # =========================================================
    MAP_MODE: str = "radial"   # "radial" or "blocky"
    RADIAL_CENTER: Tuple[float, float] = (0.5, 0.5)
    RADIAL_FALLOFF: float = 1.6

    V_MAX: float = 1.0
    V_MIN: float = 0.45

    # =========================================================
    # Noise model
    # =========================================================
    NOISE_SIGMA_MIN: float = 0.01
    NOISE_SIGMA_ALPHA: float = 0.08

    # =========================================================
    # Region thresholds
    # Region 0: high-quality
    # Region 1: intermediate
    # Region 2: degraded
    # =========================================================
    THRESH_HIGH: float = 0.78
    THRESH_MID: float = 0.58

    # =========================================================
    # Kernel size
    # Practical choice for hardware-friendly discrete bank
    # =========================================================
    KERNEL_SIZE: int = 3

    # =========================================================
    # Global candidate search
    # =========================================================
    GLOBAL_SIGMA_CANDIDATES: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8)

    # =========================================================
    # Content map
    # content 0: flat
    # content 1: texture
    # content 2: edge
    # =========================================================
    CONTENT_EDGE_PERCENTILE: float = 85.0
    CONTENT_TEXTURE_PERCENTILE: float = 55.0
    CONTENT_LOCAL_VAR_WIN: int = 5
    CONTENT_PRE_SMOOTH_SIGMA: float = 0.6
    CONTENT_MEDIAN_SIZE: int = 5

    # =========================================================
    # Selection score weights
    # normalized on validation candidates
    # =========================================================
    SCORE_WEIGHTS: Dict[str, float] = field(
        default_factory=lambda: {
            "mse": 0.40,
            "grad": 0.25,
            "edge_mse": 0.25,
            "region_mse": 0.10,
        }
    )

    EDGE_PERCENTILE_FOR_SCORE: float = 75.0

    # =========================================================
    # Content-aware PDK presets
    # Each LUT entry: (region, content) -> kernel spec
    # kernel spec examples:
    # {"family": "gaussian", "sigma": 1.0}
    # {"family": "binomial"}
    # {"family": "unsharp", "sigma": 0.8, "amount": 0.5}
    # {"family": "dog_highboost", "sigma_small": 0.7, "sigma_large": 1.4, "amount": 0.25}
    # =========================================================
    RULE_PRESETS: List[dict] = field(
        default_factory=lambda: [
            {
                "name": "gaussian_only_9state",
                "lut": {
                    (0, 0): {"family": "gaussian", "sigma": 0.6},
                    (0, 1): {"family": "gaussian", "sigma": 0.5},
                    (0, 2): {"family": "gaussian", "sigma": 0.4},

                    (1, 0): {"family": "gaussian", "sigma": 1.1},
                    (1, 1): {"family": "gaussian", "sigma": 0.8},
                    (1, 2): {"family": "gaussian", "sigma": 0.6},

                    (2, 0): {"family": "gaussian", "sigma": 1.8},
                    (2, 1): {"family": "gaussian", "sigma": 1.2},
                    (2, 2): {"family": "gaussian", "sigma": 0.9},
                },
            },
            {
                "name": "mixed_practical_v1",
                "lut": {
                    (0, 0): {"family": "binomial"},
                    (0, 1): {"family": "gaussian", "sigma": 0.6},
                    (0, 2): {"family": "unsharp", "sigma": 0.8, "amount": 0.45},

                    (1, 0): {"family": "gaussian", "sigma": 1.0},
                    (1, 1): {"family": "binomial"},
                    (1, 2): {"family": "dog_highboost", "sigma_small": 0.7, "sigma_large": 1.4, "amount": 0.20},

                    (2, 0): {"family": "gaussian", "sigma": 1.8},
                    (2, 1): {"family": "gaussian", "sigma": 1.2},
                    (2, 2): {"family": "gaussian", "sigma": 0.9},
                },
            },
            {
                "name": "mixed_practical_v2",
                "lut": {
                    (0, 0): {"family": "gaussian", "sigma": 0.7},
                    (0, 1): {"family": "binomial"},
                    (0, 2): {"family": "unsharp", "sigma": 0.9, "amount": 0.55},

                    (1, 0): {"family": "gaussian", "sigma": 1.2},
                    (1, 1): {"family": "gaussian", "sigma": 0.9},
                    (1, 2): {"family": "dog_highboost", "sigma_small": 0.8, "sigma_large": 1.6, "amount": 0.18},

                    (2, 0): {"family": "gaussian", "sigma": 2.0},
                    (2, 1): {"family": "binomial"},
                    (2, 2): {"family": "gaussian", "sigma": 1.0},
                },
            },
        ]
    )

    # =========================================================
    # Visualization / save
    # =========================================================
    SAVE_INDIVIDUAL_IMAGES: bool = True
    SAVE_PANELS: bool = True
    SAVE_SUMMARY_PLOTS: bool = True


CFG = Config()
os.makedirs(CFG.SAVE_ROOT, exist_ok=True)