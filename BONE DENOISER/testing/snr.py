# Authored by Alessandra Blasioli
import os
import csv
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pyplot as plt

clean_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/data/ac"      
data_dir = "/home/ms_ablasioli/alessandra/denoiser_backup_bcn/inference2/demucsdouble_best.th"

output_csv = "snr_results.csv"

snr_noisy = []
snr_enhanced = []
file_names = []

target_fs = 16000 

def compute_snr(clean, test):
    """Compute Signal-to-Noise Ratio (SNR) between clean and test signals"""
    noise = clean - test
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')  # Infinite SNR if no noise
    return 10 * np.log10(signal_power / noise_power)

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
            score_noisy = compute_snr(clean, noisy)
            score_enhanced = compute_snr(clean, enhanced)
        except Exception as e:
            print(f"Error computing SNR for {parts}: {e}")
            continue

        print(f"{parts}: SNR Noisy = {score_noisy:.2f} dB, SNR Enhanced = {score_enhanced:.2f} dB")

        snr_noisy.append(score_noisy)
        snr_enhanced.append(score_enhanced)
        file_names.append(parts)

if snr_noisy and snr_enhanced:
    avg_noisy = np.mean(snr_noisy)
    avg_enhanced = np.mean(snr_enhanced)

    print(f"\nAverage SNR Noisy: {avg_noisy:.2f} dB")
    print(f"Average SNR Enhanced: {avg_enhanced:.2f} dB")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "SNR Noisy (dB)", "SNR Enhanced (dB)", "Enhanced - Noisy (dB)"])
        for name, n_score, e_score in zip(file_names, snr_noisy, snr_enhanced):
            writer.writerow([name, f"{n_score:.3f}", f"{e_score:.3f}", f"{e_score - n_score:.3f}"])
        writer.writerow([])
        writer.writerow(["Average", f"{avg_noisy:.3f}", f"{avg_enhanced:.3f}", f"{(avg_enhanced - avg_noisy):.3f}"])

    print(f"\nResults saved to {output_csv}")

    plt.figure(figsize=(10, 6))
    plt.hist(snr_noisy, bins=20, alpha=0.6, label="Noisy", color="salmon", edgecolor="black")
    plt.hist(snr_enhanced, bins=20, alpha=0.6, label="Enhanced", color="skyblue", edgecolor="black")
    plt.title("SNR Distribution: Noisy vs Enhanced")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Number of Samples")
    plt.legend()

    plt.savefig("snr_comparison_histogram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(snr_noisy, snr_enhanced, alpha=0.7, color="purple")
    plt.plot([min(snr_noisy), max(snr_noisy)], [min(snr_noisy), max(snr_noisy)], 'r--', label="y=x")
    plt.title("SNR Noisy vs SNR Enhanced")
    plt.xlabel("SNR Noisy (dB)")
    plt.ylabel("SNR Enhanced (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("snr_scatter.png", dpi=300)
    plt.close()

    print("Plots saved: snr_comparison_histogram.png, snr_scatter.png")
else:
    print("No valid files processed. Check file naming and paths.")
