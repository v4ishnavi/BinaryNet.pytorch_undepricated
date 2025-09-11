#!/bin/bash

# Binary ResNet Hinge Loss Experiment Runner
# Tests 5 precision configurations with Hinge Loss only - 20 epochs

# Configuration
WANDB_PROJECT=${1:-"BNN-Hinge-Loss-Precision-Study"}
DATASET="cifar10"
EPOCHS=40
BATCH_SIZE=512

echo "======================================================================"
echo "Binary ResNet Hinge Loss - Precision Configuration Study"
echo "Project: $WANDB_PROJECT"
echo "Dataset: $DATASET | Epochs: $EPOCHS | Batch Size: $BATCH_SIZE"
echo "Loss Function: Hinge Loss Only"
echo "======================================================================"

# Check if conda environment exists
if ! conda info --envs | grep -q "precog"; then
    echo "Error: Conda environment 'precog' not found!"
    echo "Please create the environment first or modify the script."
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate precog

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'precog'"
    exit 1
fi

echo "Conda environment 'precog' activated"
echo ""

# Array of 5 configurations: [name, first_layer_fp, last_layer_fp, description]
declare -a configs=(
    "all-binary:false:false:Fully Binary (first + last layers binarized)"
    "fp-first:true:false:FP First Layer (last layer binarized)"  
    "fp-last:false:true:FP Last Layer (first layer binarized)"
    "fp-both:true:true:FP First + Last Layers (middle layers binarized)"
    "regular:na:na:Regular ResNet Baseline (all layers FP32)"
)

total_configs=${#configs[@]}
start_time=$(date +%s)

echo "Starting Binary ResNet Hinge Loss experiments..."
echo "Testing $total_configs precision configurations"
echo ""

for i in "${!configs[@]}"; do
    config_num=$((i + 1))
    IFS=':' read -r config_name first_fp last_fp description <<< "${configs[$i]}"
    
    echo "[$config_num/$total_configs] $description"
    echo "----------------------------------------"
    
    config_start=$(date +%s)
    
    if [ "$config_name" = "regular" ]; then
        # Regular ResNet baseline
        run_name="hinge_regular_resnet_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
        echo "  Running Regular ResNet + CrossEntropy (baseline)..."
        
        python main_binary.py \
            --model resnet \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "$run_name" \
            --results_dir "./results" \
            --save "$run_name" \
            --print-freq 50
    else
        # Binary ResNet configurations with Hinge Loss
        precision_flags=""
        if [ "$first_fp" = "true" ]; then
            precision_flags="$precision_flags --full-precision-first"
        fi
        if [ "$last_fp" = "true" ]; then
            precision_flags="$precision_flags --full-precision-last"
        fi
        
        run_name="hinge_binary_${config_name}_${DATASET}_e${EPOCHS}_bs${BATCH_SIZE}"
        echo "  Running Binary ResNet + Hinge Loss..."
        echo "  Precision flags: $precision_flags"
        
        python main_binary_hinge.py \
            --model resnet_binary \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            $precision_flags \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-run-name "$run_name" \
            --results_dir "./results" \
            --save "$run_name" \
            --print-freq 50
    fi
    
    exit_code=$?
    config_end=$(date +%s)
    config_duration=$((config_end - config_start))
    
    if [ $exit_code -eq 0 ]; then
        echo "  Configuration '$config_name' completed in ${config_duration}s"
    else
        echo "  Configuration '$config_name' failed with exit code $exit_code"
    fi
    echo ""
done

# All experiments completed in the main loop

# Calculate total time
end_time=$(date +%s)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))
seconds=$((total_duration % 60))

echo ""
echo "======================================================================"
echo "Binary ResNet Hinge Loss Study Completed!"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved in: ./results/"
echo "WandB project: $WANDB_PROJECT"
echo ""
echo "Configurations tested (all with Hinge Loss):"
echo "  1. Fully Binary ResNet"
echo "  2. Binary + FP First Layer"  
echo "  3. Binary + FP Last Layer"
echo "  4. Binary + FP First + Last Layers"
echo "  5. Regular ResNet (CrossEntropy baseline)"
echo ""
echo "Total experiments: $total_configs"
echo "Focus: Impact of full precision layers on Binary ResNet + Hinge Loss"
echo "======================================================================"
