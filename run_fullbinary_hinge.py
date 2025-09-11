#!/usr/bin/env python3

"""
Single Binary ResNet Experiment Runner
Runs fully binary configuration with Hinge loss for 20 epochs
"""

import subprocess
import sys
import os

def run_fullbinary_hinge():
    """Run fully binary ResNet with Hinge loss"""
    
    # Configuration
    wandb_project = "inflate1_2"
    wandb_run_name = "fullbinary_hinge_epochs75"
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
        "--wandb-project", wandb_project,
        "--wandb-run-name", wandb_run_name,
        "--results_dir", "./results",
        "--save", wandb_run_name,
        "--print-freq", "50"
    ]
    
    # Run the experiment
    result = subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_fullbinary_hinge()
