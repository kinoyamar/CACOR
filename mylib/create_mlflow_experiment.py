'''
Creates a Mlflow experiment.
Does nothing when there already exists an experiment with the specified name.

Run as,
python create_experiment.py {experiment_name} [--mlruns_dir {path/to/mlruns}]
'''

import argparse
from mlflow.tracking import MlflowClient


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', action='store', type=str,
                        help='Name of a Mlflow experiment to create.')
    parser.add_argument('--mlruns_dir', action='store', type=str, default=None,
                        help='Path to "mlruns" directory.')
    return parser.parse_args()

def main():
    args = get_args()

    client = MlflowClient(tracking_uri=args.mlruns_dir)
    try:
        experiment_id = client.create_experiment(args.experiment_name)
    except Exception as e:
        print(f'Something went wrong while creating "{args.experiment_name}".')
        print('It may already exist.')
        print(e)
    else:
        print(f'Created "{args.experiment_name}". Experiment ID is {experiment_id}.')


if __name__ == '__main__':
    main()
