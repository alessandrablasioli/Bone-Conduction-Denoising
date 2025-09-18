python train.py \
  dset=vibravox \
  demucs.causal=1 \
  demucs.hidden=48 \
  bandmask=0.2 \
  demucs.resample=4 \
  remix=1 \
  shift=8000 \
  shift_same=True \
  stft_loss=True \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  segment=4.5 \
  stride=0.5 \
  ddp=1 $@