# Continual Learning of LSTM using Ant Colony Optimization

Source code for the paper:

> R. Kinoyama, N. A. V. Suryanarayanan, and H. Iba,
> “Continual learning of LSTM using ant colony optimization,”
> *2023 IEEE Congress on Evolutionary Computation (CEC)*, 2023.

Most files in `"clutils/"` and `"experiments/"` originate from [https://github.com/AndreaCossu/ContinualLearning_RecurrentNetworks](https://github.com/AndreaCossu/ContinualLearning_RecurrentNetworks) and are modified by Rikitaka Kinoyama.
Other files are made by Rikitaka Kinoyama.
Also see `"LICENSE"`.

## Requirements

```shell
pip install -r requirements.txt
```

## Usage

Run codes from the root directory of this repo (`"cl-aco/"`).
You probably need to add this directory to enviroment variable `PYTHONPATH` before execution.
You need to download the datasets on your own and place them in a specific directory before executing the codes.
(Default dataset directory is `"~/dataset"`, which you can change from config files -> `dataroot`.)

### Run an experiment using config file

Example:

```shell
python run/run_experiment.py configs/smnist/aco.yaml
```

For more information, `python run/run_experiment.py -h`

### Run an experiment multiple times using the same config file

You need 'slurm' installed system.
You need 'singularity' installed and simg file placed at `'~/simg/pytorch.simg'`.
You can create the simg file using `'cl-aco/pytorch.def'`.
Slurm submission script, which is `'cl-aco/run/run_with_singularity.sh'` must be placed at `'~/simg/run_with_singularity.sh'`.
(You might have to change the SBATCH options depending on your environment.)

Example:

```shell
python run/multi_run.py configs/smnist/aco.yaml 0 1
```

For more information, `python run/multi_run.py -h`
