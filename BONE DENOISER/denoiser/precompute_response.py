# Authored by Alessandra Blasioli
import os
from bone_fil_2 import compute_and_save_response

if __name__ == "__main__":
    folder_clean_air = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/clean_air_train"
    folder_clean_bone = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/clean_bone_train"
    out_path = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/response.npy"

    compute_and_save_response(
        folder_clean_air,
        folder_clean_bone,
        out_path=out_path,
        T=5,
        seg_len_mic=640,
        overlap_mic=320,
        rate=16000,
        segment=0.83,
        stride=0.83
    )
