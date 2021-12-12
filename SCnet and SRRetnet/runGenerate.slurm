#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:2080ti:1
#SBATCH -o LOG-%j

LOCATION=notchpeak
NAME=GenerateCCM.py
DATADIR=/uufs/chpc.utah.edu/common/home/u0967837/cycle-consistency/Neuron
DATASETDIR=Datasets

# Modules to load
module use $HOME/MyModules

# Module purge
ml cuda/10.0 cudnn
ml miniconda3/latest
ml gcc mpich
mpirun -np $SLURM_NTASKS python $DATADIR/$NAME
# https://www.chpc.utah.edu/documentation/software/python-anaconda.php