#!/bin/bash
#SBATCH --job-name=hmsc-hpc_fit
#SBATCH --account=project_462000235
#SBATCH --output=stdout/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --time=05:00:00 --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --array=509
##31-39,51-59,71-79

ind=$SLURM_ARRAY_TASK_ID

MT=${1:-2}
SAM=${2:-10}
THIN=${3:-1}
PROFILE=${4:-0}
PRECISION=${5:-64}

module use /appl/local/csc/modulefiles/
module load tensorflow/2.12
export PYTHONPATH=$PWD/../..:$PYTHONPATH
hostname

modelTypeStringSuffices=("ns" "fu" "pg" "nn" "ph")
modelTypeString="$MT${modelTypeStringSuffices[$MT]}"

nsVec=(10 20 40 80 160 320 622)
nyVec=(100 200 400 800 1600 3200 6400 12800 25955 51910 103820 207640)

ns=${nsVec[$(($ind / 100 - 1))]}
ny=${nyVec[$(($ind % 100 - 1))]}

nChains=8

data_path="/scratch/project_462000235/gtikhono/lumiproj_2022.06.03_HPC_development/examples/big_spatial"

input_path=$data_path/$(printf "init/init_%s_ns%.3d_ny%.5d_chain%.2d.rds" $modelTypeString $ns $ny $nChains)
#output_path=$data_path/$(printf "fmTF/TF_%s_ns%.3d_ny%.5d_chain%.2d_sam%.4d_thin%.4d.rds" $modelTypeString $ns $ny $nChains $SAM $THIN)
output_path=$PWD/output/$(printf "post_chain%.2d_file.rds" $SLURM_ARRAY_TASK_ID)

srun python3 -m hmsc.run_gibbs_sampler --input $input_path --output $output_path --samples $SAM --transient $(($SAM*$THIN)) --thin $THIN --verbose 100 --fse 0 --profile $PROFILE --fp $PRECISION



#module load python-data
#export PYTHONPATH="${PYTHONPATH}:./.local/lib/python3.9/site-packages"
#./mconfig --without-suid
#pip install --user --upgrade tensorflow-probability==0.20.0
#pip install --user --upgrade pyreadr
