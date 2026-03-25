import os
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class Config:
    # ---------------------------------------------------------
    # Common
    # ---------------------------------------------------------
    SEED: int = 42
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
    SAVE_ROOT: str = "../res/global_first_pdk"

    # ---------------------------------------------------------
    # Common PDK
    # zone order: 0=attention, 1=intermediate, 2=around
    # ---------------------------------------------------------
    KERNEL_SIZE: int = 3
    ZONE_KERNEL_SPECS: Dict[int, dict] = field(
        default_factory=lambda: {
            0: {"family": "unsharp", "sigma": 0.5, "amount": 0.12},  # attention
            1: {"family": "binomial"},                                # intermediate
            2: {"family": "gaussian", "sigma": 0.9},                  # around
        }
    )

    # fixed global presets, no oracle search
    GLOBAL_SIGMA_SYNTHETIC: float = 0.7
    GLOBAL_SIGMA_LOWLIGHT: float = 0.7
    GLOBAL_SIGMA_CLASSIFICATION: float = 0.7

    # zone quantization by score percentiles
    SCORE_HIGH_PERCENTILE: float = 85.0
    SCORE_MID_PERCENTILE: float = 55.0

    # ---------------------------------------------------------
    # Synthetic task
    # ---------------------------------------------------------
    SYNTHETIC_ROOT: str = "../dat/BSDS300/images/test"
    SYNTHETIC_MAX_IMAGES: int | None = 30
    SYNTHETIC_MAP_MODE: str = "radial"  # radial or blocky
    SYNTHETIC_V_MIN: float = 0.45
    SYNTHETIC_V_MAX: float = 1.0
    SYNTHETIC_NOISE_SIGMA_MIN: float = 0.01
    SYNTHETIC_NOISE_SIGMA_ALPHA: float = 0.08
    SYNTHETIC_RADIAL_CENTER: Tuple[float, float] = (0.5, 0.5)
    SYNTHETIC_RADIAL_FALLOFF: float = 1.6
    SYNTHETIC_SCORE_WEIGHTS: Tuple[float, float, float] = (0.45, 0.25, 0.30)  # residual, gradient, unreliability

    # ---------------------------------------------------------
    # Low-light task
    # ---------------------------------------------------------
    LOWLIGHT_INPUT_ROOT = "../dat/LOL/eval15/low"
    LOWLIGHT_GT_ROOT = "../dat/LOL/eval15/high"
    LOWLIGHT_MAX_IMAGES: int | None = 50
    LOWLIGHT_ILLUM_SIGMA: float = 5.0
    LOWLIGHT_SCORE_WEIGHTS: Tuple[float, float, float] = (0.35, 0.20, 0.45)  # residual, gradient, lowlightness

    # ---------------------------------------------------------
    # Natural demo task
    # ---------------------------------------------------------
    NATURAL_ROOT: str = "../dat/BSDS300/images/test"
    NATURAL_MAX_IMAGES: int | None = 20
    NATURAL_SCORE_WEIGHTS: Tuple[float, float] = (0.55, 0.45)  # residual, edge

    # ---------------------------------------------------------
    # Classification task
    # ---------------------------------------------------------
    CLS_DATA_ROOT: str = "../dataset"
    CLS_DATASET: str = "CIFAR10"
    CLS_DEVICE: str = "cuda"
    CLS_BATCH_SIZE: int = 128
    CLS_EPOCHS: int = 10
    CLS_LR: float = 1e-3
    CLS_WEIGHT_DECAY: float = 1e-5
    CLS_NUM_WORKERS: int = 0

    # zone regularization for learned classification branch
    CLS_TARGET_ZONE_RATIOS: Tuple[float, float, float] = (0.15, 0.25, 0.60)
    CLS_LOSS_W_CE: float = 1.0
    CLS_LOSS_W_RATIO: float = 0.01
    CLS_LOSS_W_ENTROPY: float = 0.002

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    SAVE_MAX_PANELS: int = 8


CFG = Config()
os.makedirs(CFG.SAVE_ROOT, exist_ok=True)