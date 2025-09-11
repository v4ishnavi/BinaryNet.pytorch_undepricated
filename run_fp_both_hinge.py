#!/usr/bin/env python3

"""
Binary ResNet with Full Precision First + Last Layers + Hinge Loss
"""

import subprocess
import sys

def run_fp_both_hinge():
    """Run binary ResNet with full precision first + last layers + Hinge loss"""
    
    # Configuration
    wandb_project = "inflate1_2"
    wandb_run_name = "fp_both_hinge_epochs75"
    dataset = "cifar10"
    epochs = 75
    batch_size = 64
    model = "resnet_binary"
    
    # Build command
    cmd = [
        "python", "main_binary_hinge.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--full-precision-first",
        "--full-precision-last",
        "--wandb-project", wandb_project,
        "--wandb-run-name", wandb_run_name,
        "--results_dir", "./results",
        "--save", wandb_run_name,
        "--print-freq", "50"
    ]
    
    # Run the experiment
    result = subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_fp_both_hinge()
