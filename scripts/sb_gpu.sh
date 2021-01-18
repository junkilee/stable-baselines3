#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH -t 00:05:00

# Load a CUDA module
module load cuda

# Run program
./my_cuda_program
