#!/bin/bash

#SBATCH --job-name=pytorch_train
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=pytorch_train.%j.out

module purge
# module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.0.1
module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 torchvision/0.16.0-CUDA-12.1.1

echo "===== Job Info ====="
echo "Job ID: $SLURM_JOBID"
echo "Running on: $(hostname)"

echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

echo "===== GPU Info (nvidia-smi) ====="
nvidia-smi || echo "No GPU detected"

python train.py