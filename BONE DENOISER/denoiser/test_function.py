# Authored by Alessandra Blasioli

import os
import soundfile as sf
import torchaudio
import torch
from bone_fil_2 import build_bc_from_waveform_ac
folder_ac = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/valentini/clean_test/clean_testset_wav"
n_test = 10  
out_dir = "bc_debug"

os.makedirs(out_dir, exist_ok=True)
ac_files = sorted([f for f in os.listdir(folder_ac) if f.endswith(".wav")])[:n_test]

for i, fname in enumerate(ac_files):
    path_ac = os.path.join(folder_ac, fname)

    wave_ac, sr = torchaudio.load(path_ac)
    wave_ac = wave_ac.squeeze(0)

    bc_wave = build_bc_from_waveform_ac(wave_ac, rate=sr)

    bc_wave_np = bc_wave.squeeze(0).cpu().numpy().astype("float32")
    out_path = os.path.join(out_dir, f"bc_test_{i}.wav")
    sf.write(out_path, bc_wave_np, sr)
    print(f"Saved {out_path}")

print("Done! 10 BC test files saved in:", out_dir)
