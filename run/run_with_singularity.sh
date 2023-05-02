#!/bin/bash

#SBATCH -p p,v
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mail-user xxx@xxx
#SBATCH --mail-type END,FAIL

SINGULARITYENV_PYTHONPATH="$(pwd)"
export SINGULARITYENV_PYTHONPATH

srun singularity exec --nv ~/simg/pytorch.simg python "$@"
