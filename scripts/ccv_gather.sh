#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 15:00:00

module load python/3.7.4
module load anaconda/2020.02
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh

source activate sb3

echo "current directory - "$SLURM_SUBMIT_DIR
echo "current conda environment - "$CONDA_DEFAULT_ENV

conda info

# --config-path 
/users/jl84/anaconda/sb3/bin/python -c "import sys; print('\n'.join(sys.path))"
/users/jl84/anaconda/sb3/bin/python -m experiment.handlers --config-name load_ccv
