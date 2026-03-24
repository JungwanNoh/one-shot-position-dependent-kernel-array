import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Config:
    DATA_ROOT: str = "../dat/BSDS300/images/test"
    SAVE_ROOT: str = "../res/variability_simulation"

    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    MAX_IMAGES: int = 30
    FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    SEED: int = 42

    MAP_MODE: str = "radial"   # "radial" or "blocky"
    RADIAL_CENTER: Tuple[float, float] = (0.5, 0.5)
    RADIAL_FALLOFF: float = 1.6

    V_MAX: float = 1.0
    V_MIN: float = 0.45

    NOISE_SIGMA_MIN: float = 0.01
    NOISE_SIGMA_ALPHA: float = 0.08

    THRESH_HIGH: float = 0.78
    THRESH_MID: float = 0.58

    KERNEL_SIZE: int = 3
    GLOBAL_SIGMA: float = 1.0
    GLOBAL_SIGMA_CANDIDATES: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2)

    REGION_SIGMAS: Dict[int, float] = field(
        default_factory=lambda: {
            0: 0.5,
            1: 1.0,
            2: 1.8,
        }
    )

    SAVE_INDIVIDUAL_IMAGES: bool = True
    SAVE_PANELS: bool = True
    SAVE_SUMMARY_PLOTS: bool = True


CFG = Config()
os.makedirs(CFG.SAVE_ROOT, exist_ok=True)