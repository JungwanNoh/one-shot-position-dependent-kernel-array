import os
from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class Config:
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    TRAIN_ROOT: str = "../dat/BSDS300/images/train"
    TEST_ROOT: str = "../dat/BSDS300/images/test"
    SAVE_ROOT: str = "../res/nn_parameterized_foveated_zoning"

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    MAX_TRAIN_IMAGES: int | None = None
    MAX_TEST_IMAGES: int | None = None
    VAL_RATIO: float = 0.15
    FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    SEED: int = 42
    DEVICE: str = "cuda"
    BATCH_SIZE: int = 8
    EPOCHS: int = 10
    LR: float = 1e-3
    NUM_WORKERS: int = 0
    WEIGHT_DECAY: float = 1e-5

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
    # Kernel setup (3x3)
    # zone order: 0=attention, 1=intermediate, 2=around
    # ---------------------------------------------------------
    KERNEL_SIZE: int = 3
    GLOBAL_SIGMA_CANDIDATES: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.1)

    ZONE_KERNEL_SPECS: Dict[int, dict] = field(
        default_factory=lambda: {
            0: {"family": "unsharp", "sigma": 0.5, "amount": 0.12},  # attention
            1: {"family": "binomial"},                                # intermediate
            2: {"family": "gaussian", "sigma": 0.9},                  # around
        }
    )

    # ---------------------------------------------------------
    # Foveated parameter ranges
    # ---------------------------------------------------------
    R1_MIN: float = 0.05
    R1_MAX: float = 0.18
    DR_MIN: float = 0.08
    DR_MAX: float = 0.18

    SOFT_TEMP: float = 0.025

    # heatmap -> soft-argmax temperature
    HEATMAP_SOFTMAX_TEMP: float = 0.07

    # weak zone-ratio prior only
    TARGET_ZONE_RATIOS: Tuple[float, float, float] = (0.12, 0.23, 0.65)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    LOSS_W_L1: float = 0.70
    LOSS_W_GRAD: float = 0.30
    LOSS_W_RATIO: float = 0.01
    LOSS_W_ENTROPY: float = 0.002

    # model/global selection
    SELECT_W_L1: float = 0.70
    SELECT_W_GRAD: float = 0.30

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    SAVE_PANELS: bool = True
    SAVE_INDIVIDUAL_IMAGES: bool = True
    SAVE_MAX_PANELS: int = 8


CFG = Config()
os.makedirs(CFG.SAVE_ROOT, exist_ok=True)