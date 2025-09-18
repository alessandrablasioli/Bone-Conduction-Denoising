#!/bin/bash

# Define paths to Vibravox datasets
noisy_train=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_air_train_bcn1
clean_train=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/clean_air_train
bone_train=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_bone_train_bcn1
noisy_test=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_air_test_bcn
clean_test=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/clean_air_test
bone_test=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_bone_test_bcn

noisy_dev=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_air_validation_bcn
clean_dev=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/clean_air_validation
bone_dev=/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/noise_bone_validation_bcn


# Create necessary directories
mkdir -p egs/vibravox/tr
mkdir -p egs/vibravox/cv
mkdir -p egs/vibravox/tt

# Process training set
python -m denoiser.audio $noisy_train > egs/vibravox/tr/noisy.json
python -m denoiser.audio $clean_train > egs/vibravox/tr/clean.json
python -m denoiser.audio $bone_train > egs/vibravox/tr/bone.json


# Process test set
python -m denoiser.audio $noisy_test > egs/vibravox/tt/noisy.json
python -m denoiser.audio $clean_test > egs/vibravox/tt/clean.json
python -m denoiser.audio $bone_test > egs/vibravox/tt/bone.json


# Process dev (validation) set
python -m denoiser.audio $noisy_dev > egs/vibravox/cv/noisy.json
python -m denoiser.audio $clean_dev > egs/vibravox/cv/clean.json
python -m denoiser.audio $bone_dev > egs/vibravox/cv/bone.json