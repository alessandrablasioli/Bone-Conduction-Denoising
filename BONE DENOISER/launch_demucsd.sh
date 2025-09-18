CUDA_VISIBLE_DEVICES=2,3 python train.py \
  dset=valentini \
  demucsdouble.causal=1 \
  demucsdouble.hidden=48 \
  bandmask=0.2 \
  remix=1 \
  shift=8000 \
  shift_same=True \
  stft_loss=True \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  segment=4.5 \
  stride=0.5 \
  ddp=1 $@