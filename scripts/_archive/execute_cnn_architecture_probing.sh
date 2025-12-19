#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

models=(
    "tf_efficientnet_b0_ns"
    "tf_efficientnet_b1_ns"
    "tf_efficientnet_b2_ns"
    "tf_efficientnet_b3_ns"
    "resnet18"
    "resnet34"
    "resnet50"
    "resnet101"
    "inception_v3"
    "inception_v4"
)

for model in "${models[@]}"
do
    python "$SCRIPT_DIR/cnn_architecture_probing.py" \
        --model_name "$model" \
        --epochs 5 \
        --batch_size 32 \
        --lr 1e-3 \
        --vote_method max_vote_window \
        --seed 42 \
        --num_workers 8 \
        --wandb \
        --wandb_project hms-aicomp-multispec-cnn-architectures
    echo "Completed $model"
done
