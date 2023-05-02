import os
import sys
import argparse
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME, MLFLOW_GIT_COMMIT


def get_main_file():
    if len(sys.argv) > 1:
        return sys.argv[0]
    return None

# Mainly adopted from,
# https://github.com/mlflow/mlflow/blob/55a027967fb496d2970ab40b8333b266a3746211/mlflow/tracking/context/git_context.py#L11
def get_git_commit_hash():
    try:
        import git
    except ImportError as e:
        print('Failed to import package "git".')
        print(e)
        return None
    try:
        path = os.path.dirname(get_main_file())
        repo = git.Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except Exception as e:
        print('Failed to get commit info.')
        print(e)
        return None

class MlflowParser(argparse.ArgumentParser):
    '''
    Base ArgumentParser for Mlflow experiments.
    '''

    def __init__(self):
        super().__init__()
        self.add_argument('--name', '-n', action='store', type=str, default='Default',
                          help='Experiment name.')
        self.add_argument('--mlruns-dir', action='store', type=str, default=None,
                          help='Path to "mlruns" directory.')

def add_mlflow_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--name', '-n', action='store', type=str, default='Default',
                        help='Experiment name.')
    parser.add_argument('--mlruns-dir', action='store', type=str, default=None,
                        help='Path to "mlruns" directory.')

# Mostly adopted from https://zenn.dev/matsutakk/articles/75ef57fcd67334
class MlflowLogger():
    '''
    Wrapper class for MlflowClient to log mlflow runs more easily.
    '''

    def __init__(self, experiment_name, mlruns_dir=None):
        self.client = MlflowClient(tracking_uri=mlruns_dir)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        finally:
            self.experiment = self.client.get_experiment(self.experiment_id)

        self.run = None

    def start_run(self, run_name=None):
        if self.run is None:
            self._create_run(run_name)
            print(f'New run started. ({self.run_id})')
        else:
            print(f'There is already an active run. ({self.run_id})')

    def _create_run(self, run_name=None):
        tags = {}
        if run_name is not None:
            tags[MLFLOW_RUN_NAME] = str(run_name)
        main_file = get_main_file()
        if main_file is not None:
            tags[MLFLOW_SOURCE_NAME] = main_file
        commit = get_git_commit_hash()
        if commit is not None:
            tags[MLFLOW_GIT_COMMIT] = commit

        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id

    def end_run(self):
        if self.run is not None:
            self.client.set_terminated(self.run_id)
            self.run = None
            print(f'Run ended. ({self.run_id})')
        else:
            print('There is no active run to end.')
        
    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_params(self, params):
        for key, value in params.items():
            self.log_param(key, str(value))

    def log_metric(self, key, value, step=None):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path=artifact_path)
