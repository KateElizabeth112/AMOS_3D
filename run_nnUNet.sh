#!/bin/bash

# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/AMOS_3D/'
ROOT_DIR='/vol/biomedic3/kc2322/data/AMOS_3D'

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Run script to generate dataset json
#python3 generateDatasetJson.py -r $ROOT_DIR -n $DS -t $TASK

# Plan and preprocess data
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity

# Train
#nnUNetv2_train 200 3d_fullres 0