import os

SEED = 42
FILE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SAVE_ROOT = '../res/global_first_pdk_branch'

KERNEL_SIZE = 3
ZONE_KERNEL_SPECS = {
    0: {'family': 'unsharp', 'sigma': 0.5, 'amount': 0.12},
    1: {'family': 'binomial'},
    2: {'family': 'gaussian', 'sigma': 0.9},
}
GLOBAL_SIGMA = {
    'synthetic': 0.7,
    'lowlight': 0.7,
    'natural': 0.7,
    'classification': 0.7,
}
ZONE = {
    'score_blur_sigma': 1.0,
    'high_percentile': 88.0,
    'mid_percentile': 60.0,
    'min_component_area': 64,
    'dilation_iters': 3,
}
SYNTHETIC = {
    'root': '../dat/BSDS300/images/test',
    'max_images': 30,
    'image_size': (256, 256),
    'map_mode': 'radial',
    'vmin': 0.45,
    'vmax': 1.0,
    'noise_sigma_min': 0.01,
    'noise_sigma_alpha': 0.08,
    'radial_center': (0.5, 0.5),
    'radial_falloff': 1.6,
    'score_weights': {'residual': 0.45, 'gradient': 0.25, 'extra': 0.30},
}
LOWLIGHT = {
    'input_root': '../dat/LOL/eval15/low',
    'gt_root': '../dat/LOL/eval15/high',
    'max_images': None,
    'image_size': (256, 256),
    'illum_sigma': 5.0,
    'score_weights': {'residual': 0.35, 'gradient': 0.20, 'extra': 0.45},
}
NATURAL = {
    'root': '../dat/BSDS300/images/test',
    'max_images': 20,
    'image_size': (256, 256),
    'score_weights': {'residual': 0.55, 'gradient': 0.45},
}
CLASSIFICATION = {
    'dataset': 'TINYIMAGENET',
    'data_root': '../dat/torchvision',
    'tiny_imagenet_root': '../dat/tiny-imagenet-200',
    'device': 'cuda',
    'batch_size': 128,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'num_workers': 0,
    'save_root': '../res/global_first_pdk_classification',
    'score_weights': {'residual': 0.55, 'gradient': 0.45},
    'zone_high_percentile': 88.0,
    'zone_mid_percentile': 60.0,
}

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(CLASSIFICATION['save_root'], exist_ok=True)
