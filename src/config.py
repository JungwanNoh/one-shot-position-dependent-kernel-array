import os

SEED = 42
FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SAVE_ROOT = '../res/01_glovalvspdk_test'
KERNEL_SIZE = 3

# Zone id semantics used across all tasks.
# 0: preserve   -> keep structurally important regions minimally changed
# 1: recover    -> selectively boost candidate edges weakened by the global baseline
# 2: suppress   -> smooth regions dominated by noise / non-edge content
TASK_ZONE_KERNEL_SPECS = {
    'synthetic': {
        0: {'family': 'identity'},
        1: {'family': 'highboost', 'alpha': 1.00},
        2: {'family': 'gaussian', 'sigma': 0.95},
    },
    'lowlight': {
        0: {'family': 'identity'},
        1: {'family': 'highboost', 'alpha': 0.70},
        2: {'family': 'binomial'},
    },
    'natural': {
        0: {'family': 'identity'},
        1: {'family': 'highboost', 'alpha': 0.85},
        2: {'family': 'binomial'},
    },
}

# Single fixed global kernel per task.
# Intentionally stationary and smoothing-oriented to expose the single-kernel trade-off.
GLOBAL_KERNEL_SPECS = {
    'synthetic': {'family': 'binomial'},
    'lowlight': {'family': 'binomial'},
    'natural': {'family': 'binomial'},
}

ZONE_LABELS = {
    0: 'preserve',
    1: 'recover',
    2: 'suppress',
}

ZONE = {
    'score_blur_sigma': 1.0,
}

SYNTHETIC = {
    'root': '../dat/BSDS300/images/test',
    'max_images': 30,
    'image_size': (256, 256),
    'noise_sigma_min': 0.00,
    'noise_sigma_max': 0.10,
    'blur_sigma_min': 0.25,
    'blur_sigma_max': 1.35,
    'corruption_field_sigma': 18.0,
    'score_weights': {'missed': 0.70, 'coherence': 0.30},
    'panel_limit': 8,
}

LOWLIGHT = {
    'input_root': '../dat/LOL/eval15/low',
    'gt_root': '../dat/LOL/eval15/high',
    'max_images': None,
    'image_size': (256, 256),
    'score_weights': {'missed': 0.45, 'coherence': 0.30, 'brightness': 0.15, 'noise': 0.25},
    'panel_limit': 8,
}

NATURAL = {
    'root': '../dat/BSDS300/images/test',
    'max_images': 20,
    'image_size': (256, 256),
    'score_weights': {'missed': 0.50, 'coherence': 0.40, 'texture_penalty': 0.20},
    'panel_limit': 8,
}

CLASSIFICATION = {
    # Dataset: 'TINYIMAGENET' | 'CIFAR10' | 'STL10'
    'dataset': 'TINYIMAGENET',
    'data_root': '../dat/torchvision',
    'tiny_imagenet_root': '../dat/tiny-imagenet-200',
    'save_dir': os.path.join(SAVE_ROOT, 'classification'),
    'batch_size': 128,
    'epochs': 10,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'backbone': 'shufflenet',         # shufflenet_v2_x0_5 (lightweight)
    # Global kernel applied before classification
    'global_sigma': 0.7,
    # PDK zone kernels (same semantic as image tasks)
    'zone_kernel_specs': {
        0: {'family': 'unsharp', 'sigma': 0.5, 'amount': 0.12},  # preserve (attention zone)
        1: {'family': 'binomial'},                                 # intermediate
        2: {'family': 'gaussian', 'sigma': 0.9},                  # suppress (background)
    },
    # Zone ratio regularization: keep zones from collapsing to one
    'ratio_target': (0.15, 0.25, 0.60),
    'loss_w_ce': 1.0,
    'loss_w_ratio': 0.01,
    'loss_w_entropy': 0.002,
}

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(CLASSIFICATION['save_dir'], exist_ok=True)
