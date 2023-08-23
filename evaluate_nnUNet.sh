#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -N nnUNet_AMOS_evaluate

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 evaluateValidationSet.py -t $1