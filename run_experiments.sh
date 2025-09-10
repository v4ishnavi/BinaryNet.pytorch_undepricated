#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate precog

# Configuration
DATASET="cifar10"
EPOCHS=50
BATCH_SIZE=512
WANDB_PROJECT=${1:-"BNN-Experiments"}
USE_FULL_PRECISION_FIRST=${2:-"false"}  # Second argument to enable full precision first layer

if [ "$USE_FULL_PRECISION_FIRST" = "true" ]; then
    FULL_PRECISION_FLAG="--full-precision-first"
    EXPERIMENT_SUFFIX="_fp_first"
    echo "=== BNN Experiments with Full Precision First Layer ($DATASET, $EPOCHS epochs) ==="
else
    FULL_PRECISION_FLAG=""
    EXPERIMENT_SUFFIX=""
    echo "=== BNN Experiments with All Binary ($DATASET, $EPOCHS epochs) ==="
fi

# 1. Regular ResNet + CrossEntropy (baseline)
REGULAR_NAME="regular_resnet_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
echo "1/4: Regular ResNet ($EPOCHS epochs)..."
python main_binary.py --model resnet --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --wandb-project "$WANDB_PROJECT" --wandb-run-name "$REGULAR_NAME" --results_dir "./results" --save "$REGULAR_NAME" --print-freq 100

# 2. Binary ResNet + CrossEntropy  
BINARY_CE_NAME="binary_resnet_ce${EXPERIMENT_SUFFIX}_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
echo "2/4: Binary ResNet CrossEntropy ($EPOCHS epochs)..."
python main_binary.py --model resnet_binary --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE $FULL_PRECISION_FLAG --wandb-project "$WANDB_PROJECT" --wandb-run-name "$BINARY_CE_NAME" --results_dir "./results" --save "$BINARY_CE_NAME" --print-freq 100

# 3. Binary ResNet + Hinge Loss
BINARY_HINGE_NAME="binary_resnet_hinge${EXPERIMENT_SUFFIX}_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
echo "3/4: Binary ResNet Hinge ($EPOCHS epochs)..."
python main_binary_hinge.py --model resnet_binary --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE $FULL_PRECISION_FLAG --wandb-project "$WANDB_PROJECT" --wandb-run-name "$BINARY_HINGE_NAME" --results_dir "./results" --save "$BINARY_HINGE_NAME" --print-freq 100

# 4. Binary ResNet + CrossEntropy + Full Precision First (if not already done)
if [ "$USE_FULL_PRECISION_FIRST" = "false" ]; then
    BINARY_CE_FP_NAME="binary_resnet_ce_fp_first_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
    echo "4/4: Binary ResNet CrossEntropy + Full Precision First ($EPOCHS epochs)..."
    python main_binary.py --model resnet_binary --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --full-precision-first --wandb-project "$WANDB_PROJECT" --wandb-run-name "$BINARY_CE_FP_NAME" --results_dir "./results" --save "$BINARY_CE_FP_NAME" --print-freq 100
else
    echo "4/4: Skipping duplicate full precision first experiment..."
fi

echo "✓ All experiments done! Results in ./results/"
