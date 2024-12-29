#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --account=su007-fl-gpu
#SBATCH --output=output_glue.log
#SBATCH --error=error_glue.log

module purge
module load GCC/13.2.0  OpenMPI/4.1.6 PIP-PyTorch/2.4.0-CUDA-12.4.0

srun python ResNet18.py