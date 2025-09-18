#!/bin/bash

MODELS_DIR="/home/ms_ablasioli/alessandra/exp_bandmask=0.2,demucsdouble.causal=1,demucsdouble.hidden=48,dset=vibravox,remix=1,segment=4.5,shift=8000,shift_same=True,stft_loss=True,stft_mag_factor=0.1,stft_sc_factor=0.1,stride=0.5/best.th"
NOISY_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/colleced_data_noisy/ac"
OUTPUT_DIR="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference"

mkdir -p "$OUTPUT_DIR"

for MODEL_PATH in "$MODELS_DIR"/*; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    echo "Enhancing model: $MODEL_NAME"
    MODEL_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_OUTPUT_DIR"

    python -m denoiser.enhance \
        --model_path="$MODEL_PATH" \
        --noisy_dir="$NOISY_DIR" \
        --out_dir="$MODEL_OUTPUT_DIR" \
        > "$MODEL_OUTPUT_DIR/enhance.log" 2>&1

    echo "File saved in $MODEL_OUTPUT_DIR"
done

echo "Enhance COMPLETED"