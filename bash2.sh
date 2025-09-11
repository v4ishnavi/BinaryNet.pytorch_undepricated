#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate patchEnv
# Binary ResNet Sweep: batch size, lr, optimizer, weight decay

WANDB_PROJECT="binary-resnet-sweep"
DATASET="cifar10"
EPOCHS=20
MODEL="resnet_binary"
INFLATE_FACTOR=5

# Sweep values
BATCH_SIZES=(64 128)
LRS=(0.001 0.005 0.01)
OPTIMIZERS=("Adam" "SGD")
WDS=(1e-4 1e-5)

echo "Starting sweep..."
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Learning rates: ${LRS[@]}"
echo "Optimizers: ${OPTIMIZERS[@]}"
echo "Weight decays: ${WDS[@]}"
echo ""

for BATCH in "${BATCH_SIZES[@]}"; do
  for LR in "${LRS[@]}"; do
    for OPT in "${OPTIMIZERS[@]}"; do
      for WD in "${WDS[@]}"; do
        run_name="bs${BATCH}_lr${LR}_${OPT}_wd${WD}_e${EPOCHS}"
        echo ">>> Running $run_name"

        python main_binary_hinge.py \
          --model $MODEL \
          --dataset $DATASET \
          --epochs $EPOCHS \
          --batch-size $BATCH \
          --lr $LR \
          --optimizer $OPT \
          --weight-decay $WD \
          --inflate $INFLATE_FACTOR \
          --wandb-project $WANDB_PROJECT \
          --wandb-run-name $run_name \
          --results_dir ./results \
          --save $run_name \
          --print-freq 100
      done
    done
  done
done

echo "Sweep finished."
