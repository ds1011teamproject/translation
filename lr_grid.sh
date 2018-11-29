#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=rfbase
#SBATCH --mail-type=END
#SBATCH --mail-user=rf1316@nyu.edu
#SBATCH --output=rfbase.txt
#SBATCH --gres=gpu:1

# Activate the conda environment
export PATH=“/home/rf1316/miniconda3/bin:$PATH”
. /home/rf1316/miniconda3/etc/profile.d/conda.sh

conda activate torch

# Run the training script
for VARIABLE in 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
do
    PYTHONPATH=$PYTHONPATH:. python lr_grid.py -c user_rf1316 -m RNN_GRU -l $VARIABLE
done
