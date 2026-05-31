#!/bin/bash
#SBATCH --job-name=hmsc_test
#SBATCH --account=project_XXXXXXX
#SBATCH --output=output/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1

# Create output directory
mkdir -p output
# Load TensorFlow module
module load tensorflow/2.18
# Activate virtual environment
source /projappl/project_XXXXXXX/hmsc_tf_env/bin/activate

# Test TensorFlow + GPU
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs available:', tf.config.list_physical_devices('GPU'))"
# Test hmsc-hpc import
python3 -c "import hmsc; print('hmsc-hpc imported successfully')"

# Example: Run hmsc Gibbs sampler
# Replace paths with your actual data
INPUT_FILE="/scratch/project_XXXXXXX/your_data/init_model.rds"
OUTPUT_FILE="/scratch/project_XXXXXXX/your_data/output_samples.rds"

srun python3 -m hmsc.run_gibbs_sampler \
  --input $INPUT_FILE \
  --output $OUTPUT_FILE \
  --samples 1000 \
  --transient 500 \
  --thin 1 \
  --verbose 100

