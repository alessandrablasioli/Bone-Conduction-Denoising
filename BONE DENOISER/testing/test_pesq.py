# Authored by Alessandra Blasioli
import os
import csv
import numpy as np
from pesq import pesq
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt

clean_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/data/ac"
data_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference2/demucsdouble_best.th"
output_csv = "pesq_results.csv"

pesq_noisy = []
pesq_enhanced = []
file_names = []

target_fs = 16000 

for filename in os.listdir(data_dir):
    if filename.endswith("_enhanced.wav"):
        parts = filename.split("_NOISE_")[0]
        clean_name = parts + ".wav"
        clean_file = os.path.join(clean_dir, clean_name)

        noisy_file = os.path.join(data_dir, filename.replace("_enhanced.wav", "_noisy.wav"))
        enhanced_file = os.path.join(data_dir, filename)

        print(f"Clean: {clean_file}")
        print(f"Noisy: {noisy_file}")
        print(f"Enhanced: {enhanced_file}")

        if not os.path.exists(clean_file):
            print(f"Missing clean file: {clean_file}")
            continue
        if not os.path.exists(noisy_file):
            print(f"Missing noisy file: {noisy_file}")
            continue

        
        fs_clean, clean = wavfile.read(clean_file)
        _, noisy = wavfile.read(noisy_file)
        _, enhanced = wavfile.read(enhanced_file)

        
        if fs_clean not in [8000, 16000]:
            print(f"Resampling clean file from {fs_clean} Hz to {target_fs} Hz: {clean_file}")
            duration = len(clean) / fs_clean
            num_samples = int(duration * target_fs)
            clean = resample(clean, num_samples)
            fs_clean = target_fs

        if len(noisy) != len(clean):
            print(f"Warning: Resampling noisy file to match clean length")
            noisy = resample(noisy, len(clean))
        if len(enhanced) != len(clean):
            print(f"Warning: Resampling enhanced file to match clean length")
            enhanced = resample(enhanced, len(clean))

        try:
            score_noisy = pesq(fs_clean, clean, noisy, 'wb')
            score_enhanced = pesq(fs_clean, clean, enhanced, 'wb')
        except Exception as e:
            print(f"Error computing PESQ for {parts}: {e}")
            continue

        print(f"{parts}: PESQ Noisy = {score_noisy:.2f}, PESQ Enhanced = {score_enhanced:.2f}")

        pesq_noisy.append(score_noisy)
        pesq_enhanced.append(score_enhanced)
        file_names.append(parts)


if pesq_noisy and pesq_enhanced:
    avg_noisy = np.mean(pesq_noisy)
    avg_enhanced = np.mean(pesq_enhanced)

    print(f"\nAverage PESQ Noisy: {avg_noisy:.2f}")
    print(f"Average PESQ Enhanced: {avg_enhanced:.2f}")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "PESQ Noisy", "PESQ Enhanced", "Enhanced - Noisy"])
        for name, n_score, e_score in zip(file_names, pesq_noisy, pesq_enhanced):
            writer.writerow([name, f"{n_score:.3f}", f"{e_score:.3f}", f"{e_score - n_score:.3f}"])
        writer.writerow([])
        writer.writerow(["Average", f"{avg_noisy:.3f}", f"{avg_enhanced:.3f}", f"{(avg_enhanced - avg_noisy):.3f}"])

    print(f"\nResults saved to {output_csv}")
    plt.figure(figsize=(10, 6))
    plt.hist(pesq_noisy, bins=20, alpha=0.6, label="Noisy", color="salmon", edgecolor="black")
    plt.hist(pesq_enhanced, bins=20, alpha=0.6, label="Enhanced", color="skyblue", edgecolor="black")
    plt.title("PESQ Distribution: Noisy vs Enhanced")
    plt.xlabel("PESQ Score")
    plt.ylabel("Number of Samples")
    plt.legend()

    plt.savefig("pesq_comparison_histogram.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(pesq_noisy, pesq_enhanced, alpha=0.7, color="purple")
    plt.plot([min(pesq_noisy), max(pesq_noisy)], [min(pesq_noisy), max(pesq_noisy)], 'r--', label="y=x")
    plt.title("PESQ Noisy vs PESQ Enhanced")
    plt.xlabel("PESQ Noisy")
    plt.ylabel("PESQ Enhanced")
    plt.legend()
    plt.grid(True)
    plt.savefig("pesq_scatter.png", dpi=300)
    plt.close()

    print("Plots saved: pesq_comparison_histogram.png, pesq_scatter.png")
else:
    print("No valid files processed. Check file naming and paths.")
