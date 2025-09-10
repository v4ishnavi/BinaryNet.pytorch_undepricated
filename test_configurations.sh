#!/bin/bash

# Quick test script for different binary ResNet configurations

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate precog

DATASET="cifar10"
EPOCHS=5  # Just for quick testing
BATCH_SIZE=512
WANDB_PROJECT="BNN-QuickTest"

echo "=== Quick Binary ResNet Configuration Test ==="

echo "Testing Binary ResNet configurations (5 epochs each for quick verification):"

# Test 1: Fully binarized (original approach)
echo ""
echo "1/3: Fully Binarized ResNet..."
python main_binary.py --model resnet_binary --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --wandb-project "$WANDB_PROJECT" --wandb-run-name "test_fully_binary" --results_dir "./test_results" --save "test_fully_binary" --print-freq 50

# Test 2: Full precision first layer
echo ""
echo "2/3: Binary ResNet with Full Precision First Layer..."
python main_binary.py --model resnet_binary --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --full-precision-first --wandb-project "$WANDB_PROJECT" --wandb-run-name "test_fp_first" --results_dir "./test_results" --save "test_fp_first" --print-freq 50

# Test 3: Regular ResNet baseline
echo ""
echo "3/3: Regular ResNet Baseline..."
python main_binary.py --model resnet --dataset $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --wandb-project "$WANDB_PROJECT" --wandb-run-name "test_regular" --results_dir "./test_results" --save "test_regular" --print-freq 50

echo ""
echo "✓ Quick test completed! Check WandB for initial performance comparison."
echo "Usage for full experiments:"
echo "  ./run_experiments.sh 'ProjectName' false    # All binary layers"
echo "  ./run_experiments.sh 'ProjectName' true     # Full precision first layer"
