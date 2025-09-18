#!/bin/bash

# Define paths to Vibravox datasets
noisy_train=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_air_train_bcn
clean_train=/home/alessandrab/denoiser_backup_bcn/vibravox/clean_air_train1
#bone_train=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_bone_train_bcn

noisy_test=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_air_test_bcn
clean_test=/home/alessandrab/denoiser_backup_bcn/vibravox/clean_air_test
#bone_test=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_bone_test_bcn

noisy_dev=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_air_validation_bcn
clean_dev=/home/alessandrab/denoiser_backup_bcn/vibravox/clean_air_validation
#bone_dev=/home/alessandrab/denoiser_backup_bcn/vibravox/noise_bone_validation_bcn


# Create necessary directories
mkdir -p egs/vibravox/tr
mkdir -p egs/vibravox/cv
mkdir -p egs/vibravox/tt

# Process training set
python -m denoiser.audio $noisy_train > egs/vibravox/tr/noisy.json
python -m denoiser.audio $clean_train > egs/vibravox/tr/clean.json
#python -m denoiser.audio $bone_train > egs/vibravox/tr/bone.json


# Process test set
python -m denoiser.audio $noisy_test > egs/vibravox/tt/noisy.json
python -m denoiser.audio $clean_test > egs/vibravox/tt/clean.json
#python -m denoiser.audio $bone_test > egs/vibravox/tt/bone.json


# Process dev (validation) set
python -m denoiser.audio $noisy_dev > egs/vibravox/cv/noisy.json
python -m denoiser.audio $clean_dev > egs/vibravox/cv/clean.json
#python -m denoiser.audio $bone_dev > egs/vibravox/cv/bone.json