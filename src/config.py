import os

SEED = 42
DEVICE = "cuda"
SAVE_ROOT = "../res/pdk_multitask"

COMMON = {
    "image_size": (256, 256),
    "kernel_size": 3,
    "zone_kernel_specs": {
        0: {"family": "unsharp", "sigma": 0.5, "amount": 0.12},  # attention
        1: {"family": "binomial"},                                  # intermediate
        2: {"family": "gaussian", "sigma": 0.9},                  # around
    },
    # realistic fixed global presets per task
    "global_specs": {
        "synthetic": {"family": "gaussian", "sigma": 0.7},
        "lowlight": {"family": "gaussian", "sigma": 0.7},
        "natural": {"family": "binomial"},
        "classification": {"family": "gaussian", "sigma": 0.7},
    },
}

SYNTHETIC = {
    "root": "../dat/BSDS300/images/test",
    "max_images": 30,
    "save_dir": os.path.join(SAVE_ROOT, "synthetic_nonuniform"),
    "map_mode": "radial",     # radial or blocky
    "radial_center": (0.5, 0.5),
    "radial_falloff": 1.6,
    "v_max": 1.0,
    "v_min": 0.45,
    "noise_sigma_min": 0.01,
    "noise_sigma_alpha": 0.08,
    # reliability -> zones, high value = good region
    "th_high": 0.78,
    "th_mid": 0.58,
    "save_max_panels": 8,
}

LOWLIGHT = {
    # paired folders with matching filenames
    "low_dir": "../dat/LOL/eval15/low",
    "high_dir": "../dat/LOL/eval15/high",
    "max_images": 50,
    "save_dir": os.path.join(SAVE_ROOT, "lowlight"),
    # bright/reliable -> attention, dark/noisy -> around
    "illum_high": 0.62,
    "illum_mid": 0.38,
    "save_max_panels": 8,
}

NATURAL = {
    "root": "../dat/BSDS300/images/test",
    "max_images": 20,
    "save_dir": os.path.join(SAVE_ROOT, "natural_demo"),
    # pseudo-fovea radii from image content center
    "r1": 0.12,
    "r2": 0.28,
    "save_max_panels": 8,
}

CLASSIFICATION = {
    # CIFAR10, STL10, TINYIMAGENET
    "dataset": "TINYIMAGENET", # "TINYIMAGENET"
    "data_root": "../dat/torchvision",
    # Standard Tiny ImageNet extracted folder, e.g. ../dat/tiny-imagenet-200
    "tiny_imagenet_root": "../dat/tiny-imagenet-200",
    "save_dir": os.path.join(SAVE_ROOT, "classification"),
    "batch_size": 128,
    # small/medium defaults; override if needed
    "epochs": 10,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "backbone": "resnet18",
    # optional stronger recipe toggles
    "use_randaugment": True,
    "randaugment_num_ops": 2,
    "randaugment_magnitude": 9,
    # PDK head regularization
    "ratio_target": (0.15, 0.25, 0.60),
    "loss_w_ce": 1.0,
    "loss_w_ratio": 0.01,
    "loss_w_entropy": 0.002,
}

os.makedirs(SAVE_ROOT, exist_ok=True)
