#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 20G # memory pool for all cores
#SBATCH --nodelist monal04 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/AMOS_3D/'
ROOT_DIR='/vol/biomedic3/kc2322/data/AMOS_3D/'

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Run script to generate dataset json
#python3 generateDatasetJson.py -r $ROOT_DIR -n $DS -t $TASK

# Plan and preprocess data
#nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity

# Train
#nnUNetv2_train 200 2d 0

# Inference
INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/Dataset200_AMOS/imagesVaSorted"
OUTPUT_FOLDER=$ROOT_ROOT_DIR"inference/preds"

echo $INPUT_FOLDER
echo $OUTPUT_FOLDER

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 200 -c 2d -f 0 --verbose