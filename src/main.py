import argparse
from utils import set_seed
from config import SEED


def run_synthetic():
    from synthetic_task import run
    run()


def run_lowlight():
    from lowlight_task import run
    run()


def run_natural():
    from natural_demo_task import run
    run()


def run_classification():
    from classification_task import run
    run()


def main():
    parser = argparse.ArgumentParser(
        description='PDK vs Global kernel comparison.\n'
                    'Tasks with GT (PSNR/SSIM available): synthetic, lowlight\n'
                    'No-GT task (qualitative only):        natural\n'
                    'Downstream proxy for natural images:  classification'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['synthetic', 'lowlight', 'natural', 'classification', 'all']
    )
    args = parser.parse_args()
    set_seed(SEED)

    if args.task == 'synthetic':
        run_synthetic()
    elif args.task == 'lowlight':
        run_lowlight()
    elif args.task == 'natural':
        run_natural()
    elif args.task == 'classification':
        run_classification()
    elif args.task == 'all':
        print('\n=== Running all tasks: synthetic -> lowlight -> natural -> classification ===')
        run_synthetic()
        run_lowlight()
        run_natural()
        run_classification()


if __name__ == '__main__':
    main()
