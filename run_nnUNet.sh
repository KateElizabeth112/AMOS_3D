#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000

# Launch virtual environment
source /rds/general/user/kc2322/home/data/AMOS_3D/venv/bin/activate

# Set environment variables
ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'
DATASET="Dataset701_Set1"

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Plan and preprocess data
nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity

# Train
nnUNetv2_train 701 3d_fullres all

# Inference
#INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/Dataset702_Set2/imagesTs"
#OUTPUT_FOLDER=$ROOT_DIR"inference/Dataset702_Set2/all"

#nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 702 -c 3d_fullres -f all -device cpu