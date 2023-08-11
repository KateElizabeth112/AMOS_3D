#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 32G # memory pool for all cores
#SBATCH --nodelist monal03 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/AMOS_3D/'
ROOT_DIR='/vol/biomedic3/kc2322/data/AMOS_3D/'
DATASET="Dataset701_Set1"

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Plan and preprocess data
#nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity

# Train
nnUNetv2_train 701 3d_fullres all

# Inference
INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/Dataset702_Set2/imagesTs"
OUTPUT_FOLDER=$ROOT_DIR"inference/Dataset702_Set2/all"

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 702 -c 3d_fullres -f all -device cpu