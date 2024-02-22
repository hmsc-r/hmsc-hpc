#!/bin/bash

#SBATCH --job-name=hmsc-hpc_fit
#SBATCH --account=project_462000235
#SBATCH --output=stdout/%A_%a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:14:59
#SBATCH --partition=small-g
#SBATCH --array=0-3

module use /appl/local/csc/modulefiles/
module load tensorflow/2.12

export PYTHONPATH=$PWD/../..:$PYTHONPATH

SAM=${1:-100}
THIN=${2:-2}

input_path=$PWD/input/init_file.rds
output_path=$PWD/output/$(printf "post_chain%.2d_file.rds" $SLURM_ARRAY_TASK_ID)

srun python3 -m hmsc.run_gibbs_sampler --input $input_path --output $output_path --samples $SAM --transient $(($SAM*$THIN)) --thin $THIN --verbose 100 --chain $SLURM_ARRAY_TASK_ID --fp 64
