import argparse

from synthetic_task import run_synthetic
from lowlight_task import run_lowlight
from natural_demo_task import run_natural_demo
from classification_task import run_classification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["synthetic", "lowlight", "natural", "classification", "all"])
    args = parser.parse_args()

    if args.task == "synthetic":
        run_synthetic()
    elif args.task == "lowlight":
        run_lowlight()
    elif args.task == "natural":
        run_natural_demo()
    elif args.task == "classification":
        run_classification()
    elif args.task == "all":
        run_synthetic()
        run_lowlight()
        run_natural_demo()
        run_classification()


if __name__ == "__main__":
    main()
