# Bone-Conduction-Denoising
Study and Application of Bone Conduction Audio for Audio Denoising Using Neural Networks


Speech denoising is a central topic in the development of modern audio devices, aiming to improve speech quality and intelligibility in noisy environments.
This thesis explores the use of Bone Conduction signals to support the training of neural networks for denoising.
Architectures operating both in the time domain and in the frequency domain were implemented, adopting a dual-input approach.
Specifically, the `DemucsDouble` network was developed for the time domain, and the `DemucsFrequencyDualLSTM` network for the frequency domain.

### Installation
 
 Clone the repo
   ```sh
   git clone https://github.com/alessandrablasioli/Bone-Conduction-Denoising.git
   ```

## Project Structure

 ```project-root/
├─ Spectrogram Study/ # Study of spectrograms of Bone Conduction audio
├─ Filter for Synthetic BC/ # Synthetic BC function (Method 1 & 2)
├─ BONE DENOISER/ # Source code of the proposed solution
├─ DENOISER/ # Validation results of Demucs
├─ Data Preparation/ # Pre-processing for Vibravox dataset
└─ README.md # This file
 ```

# Results

## Results of the different models evaluated on PESQ and STOI

| **Parameters**        | **Model**                     | **Input**                     | **PESQ** | **STOI** |
|-----------------------|--------------------------------|--------------------------------|--------:|-------:|
| lr 3e-4, 400 epochs   | BASELINE (Demucs)             | VALENTINI                      | 2.9328  | 0.94759 |
| lr 1.5e-4, 200 epochs | BASELINE (Demucs)             | VIBRAVOX (AC only)             | 2.5587  | 0.92213 |
| lr 1.5e-4, 200 epochs | Baseline in Frequency Domain  | VIBRAVOX (AC only)             | 2.3304  | 0.90261 |
| lr 1.5e-4, 200 epochs | `DemucsDouble`                | VIBRAVOX                       | 2.6965  | 0.93010 |
| lr 1.5e-4, 200 epochs | `DemucsFrequencyDualLSTM`     | VIBRAVOX                       | 2.4146  | 0.91046 |

---

## Model Improvement over Baselines

| **Model**                  | **PESQ Improvement (%)** | **STOI Improvement (%)** |
|----------------------------|------------------------:|-------------------------:|
| `DemucsDouble`             | 5.39                   | 0.86                    |
| `DemucsFrequencyDualLSTM`  | 3.61                   | 0.87                    |

## Results of the different models evaluated on PESQ and STOI with synthetic BC

| **Parameters**            | **Model**                       | **Input**                                    | **PESQ** | **STOI** |
|---------------------------|----------------------------------|-----------------------------------------------|--------:|-------:|
| lr 3e-4, 400 epochs       | BASELINE (Demucs)               | VALENTINI                                     | 2.9328  | 0.94759 |
| lr 1.5e-4, 200 epochs     | Baseline in Frequency Domain    | VALENTINI (AC only)                            | 2.7752  | 0.94141 |
| lr 1.5e-4, 200 epochs     | `DemucsDouble`                  | VALENTINI (AC + Synthetic BC METHOD 1)         | 1.9568  | 0.88828 |
| lr 1.5e-4, 200 epochs     | `DemucsFrequencyDualLSTM`       | VALENTINI (AC + Synthetic BC METHOD 1)         | 2.7510  | 0.94159 |
| lr 1.5e-4, 200 epochs     | `DemucsDouble`                  | VALENTINI (AC + Synthetic BC METHOD 2)         | 1.9617  | 0.88860 |
| lr 1.5e-4, 200 epochs     | `DemucsFrequencyDualLSTM`       | VALENTINI (AC + Synthetic BC METHOD 2)         | 2.7412  | 0.94166 |
