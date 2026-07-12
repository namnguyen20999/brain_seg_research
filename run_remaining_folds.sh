#!/bin/bash
set -uo pipefail
source /venv/main/bin/activate
export nnUNet_extTrainer=/workspace/brain_seg_research/custom_trainers
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NNUNET_ES_PATIENCE=50
export NNUNET_ES_MIN_EPOCHS=100

for fold in 1 2 3 4; do
    echo "=== STARTING FOLD $fold: $(date) ===" 
    nnUNetv2_train 1 2d "$fold" -tr nnUNetTrainerEarlyStopping > /workspace/brain_seg_research/train_fold${fold}.log 2>&1
    echo "=== FINISHED FOLD $fold: $(date) exit_code=$? ==="
done
