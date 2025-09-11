#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate precog
# Run Full Binary Hinge
python run_fullbinary_hinge.py

# Run FP First Hinge
python run_fp_first_hinge.py

# Run FP Both Hinge
python run_fp_both_hinge.py
# ./run_fp_both_hinge.py