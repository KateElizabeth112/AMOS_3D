#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=18:mem=100gb:ngpus=1:gpu_type=RTX6000
#PBS -N nnUNet_AMOS_802

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

# Set environment variables
ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'
DATASET='Dataset802_Fold3'
TASK=802

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Create dataset.json
#python3 generateDatasetJson.py -r $ROOT_DIR -n $DATASET -tc 120

# Plan and preprocess data
#nnUNetv2_plan_and_preprocess -d $TASK -c 3d_fullres -np 3 --verify_dataset_integrity

# Train
nnUNetv2_train $TASK 3d_fullres all