#!/bin/bash

MODELS_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/testing/models/bc"
NOISY_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/colleced_data_noisy/ac"
BONE_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/colleced_data_noisy/bc"
OUTPUT_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference2"

mkdir -p "$OUTPUT_DIR"

for MODEL_PATH in "$MODELS_DIR"/*; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    echo "Enhancing model: $MODEL_NAME"
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_OUTPUT_DIR"

    python -m denoiser.enhance \
        --model_path="$MODEL_PATH" \
        --noisy_dir="$NOISY_DIR" \
        --bone_dir="$BONE_DIR" \
        --out_dir="$MODEL_OUTPUT_DIR" \
        > "$MODEL_OUTPUT_DIR/enhance.log" 2>&1

    echo "File saved in $MODEL_OUTPUT_DIR"
done

echo "Enhance COMPLETED"