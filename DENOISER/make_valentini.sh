#!/bin/bash

noisy_train=/home/ms_ablasioli/alessandra/denoiser/valentini/noisy_train/noisy_trainset_28spk_wav
clean_train=/home/ms_ablasioli/alessandra/denoiser/valentini/clean_train/clean_t/clean_trainset_28spk_wav
noisy_test=/home/ms_ablasioli/alessandra/denoiser/valentini/noisy_test/noisy_t/noisy_testset_wav
clean_test=/home/ms_ablasioli/alessandra/denoiser/valentini/clean_test/clean_testset_wav
noisy_dev=/home/ms_ablasioli/alessandra/denoiser/valentini/noisy_validation
clean_dev=/home/ms_ablasioli/alessandra/denoiser/valentini/clean_validation

mkdir -p egs/val/tr
mkdir -p egs/val/cv
mkdir -p egs/val/tt

python -m denoiser.audio $noisy_train > egs/val/tr/noisy.json
python -m denoiser.audio $clean_train > egs/val/tr/clean.json

python -m denoiser.audio $noisy_test > egs/val/tt/noisy.json
python -m denoiser.audio $clean_test > egs/val/tt/clean.json

python -m denoiser.audio $noisy_dev > egs/val/cv/noisy.json
python -m denoiser.audio $clean_dev > egs/val/cv/clean.json