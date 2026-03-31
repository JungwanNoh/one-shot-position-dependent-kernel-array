import argparse
from utils import set_seed
from config import SEED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['synthetic', 'lowlight', 'natural', 'classification'])
    args = parser.parse_args()
    set_seed(SEED)
    if args.task == 'synthetic':
        from synthetic_task import run
        run()
    elif args.task == 'lowlight':
        from lowlight_task import run
        run()
    elif args.task == 'natural':
        from natural_demo_task import run
        run()
    elif args.task == 'classification':
        from classification_task import run_classification
        run_classification()

if __name__ == '__main__':
    main()
