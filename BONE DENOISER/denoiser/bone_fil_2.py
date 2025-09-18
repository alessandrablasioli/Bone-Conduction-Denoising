import librosa
import numpy as np
import scipy.signal as signal
from skimage import filters
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
from scipy.signal import butter,filtfilt, lfilter
import os


seg_len_mic = 640
overlap_mic = 320
seg_len_imu = 64
overlap_imu = 32
rate_mic = 16000
rate_imu = 16000
T = 5
segment = 0.83
stride = 0.83
freq_bin_high = int(rate_imu / rate_mic * int(seg_len_mic / 2)) + 1
time_bin = int(segment * rate_mic/(seg_len_mic-overlap_mic)) + 1
time_stride = int(stride * rate_mic/(seg_len_mic-overlap_mic))

def normalization(wave_data, rate=16000, T=5):
    b, a = signal.butter(4, 100, 'highpass', fs=rate)
    wave_data = signal.filtfilt(b, a, wave_data)
    if len(wave_data) >= T * rate:
        return wave_data[:T * rate]
    wave_data = np.pad(wave_data, (0, T * rate - len(wave_data)), 'constant', constant_values=(0, 0))
    return wave_data
def frequencydomain(wave_data, seg_len=2560, overlap=2240, rate=16000, mfcc=False):
    if mfcc:
        Zxx = librosa.feature.melspectrogram(wave_data, sr=rate, n_fft=seg_len, hop_length=seg_len-overlap, power=1)
        return Zxx, None
    else:
        f, t, Zxx = signal.stft(wave_data, nperseg=seg_len, noverlap=overlap, fs=rate)
        phase = np.exp(1j * np.angle(Zxx))
        Zxx = np.abs(Zxx)
        return Zxx, phase
def load_audio_n(name, T, seg_len=2560, overlap=2240, rate=16000, normalize=False, mfcc=False):
    wave, _ = librosa.load(name, sr=rate)
    if normalize:
        wave = normalization(wave, rate, T)
    Zxx, phase = frequencydomain(wave, seg_len=seg_len, overlap=overlap, rate=rate, mfcc=mfcc)
    return wave, Zxx, phase

def synchronization(Zxx, imu):
    in1 = np.sum(Zxx[:freq_bin_high, :], axis=0)
    in2 = np.sum(imu, axis=0)
    shift = np.argmax(signal.correlate(in1, in2)) - len(in2)
    return np.roll(imu, shift, axis=1)
def estimate_response(imu, Zxx):
    select1 = Zxx > 1 * filters.threshold_otsu(Zxx)
    select2 = imu > 1 * filters.threshold_otsu(imu)
    
    select = select2 & select1
    
    Zxx_ratio = np.divide(imu, Zxx, out=np.zeros_like(imu), where=select)
    response = np.zeros((2, freq_bin_high))
    for i in range(freq_bin_high):
        if np.sum(select[i, :]) > 0:
            response[0, i] = np.mean(Zxx_ratio[i, :], where=select[i, :])
            response[1, i] = np.std(Zxx_ratio[i, :], where=select[i, :])
    return response
def transfer_function(clip1, clip2, response):
    new_response = estimate_response(clip1, clip2)
    response = 0.25 * new_response + 0.75 * response
    return response

def filter_function(response):
    m = np.max(response)
    n1 = np.mean(response[-5:])
    n2 = np.mean(response)
    if m > 35:
        return False
    elif (2*n1) > m:
        return False
    else:
        return True

def compute_and_save_response(
    folder_clean_air,
    folder_clean_bone,
    out_path="response.npy",
    T=5,
    seg_len_mic=640,
    overlap_mic=320,
    rate=16000,
    segment=0.83,
    stride=0.83
):
    """Compute transfer function response from clean air/bone folders and save it."""
    
    air_files = sorted(os.listdir(folder_clean_air))
    bone_files = sorted(os.listdir(folder_clean_bone))
    assert len(air_files) == len(bone_files), "Mismatched clean_air / clean_bone"

    freq_bin_high = int(seg_len_mic / 2) + 1
    time_bin = int(segment * rate / (seg_len_mic - overlap_mic)) + 1
    time_stride = int(stride * rate / (seg_len_mic - overlap_mic))

    response = np.zeros((2, freq_bin_high))

    for f_air, f_bone in zip(air_files, bone_files):
        path_air = os.path.join(folder_clean_air, f_air)
        path_bone = os.path.join(folder_clean_bone, f_bone)

        _, Zxx_air_clean, _ = load_audio_n(path_air, T, seg_len_mic, overlap_mic, rate, normalize=True)
        _, Zxx_bone_clean, _ = load_audio_n(path_bone, T, seg_len_mic, overlap_mic, rate, normalize=True)

        for j in range(int((T - segment) / stride) + 1):
            clip2 = Zxx_air_clean[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
            clip1 = Zxx_bone_clean[:, j * time_stride:j * time_stride + time_bin]
            response = transfer_function(clip1, clip2, response)

    response /= len(air_files)
    np.save(out_path, response)
    print(f"Response saved to {out_path}")
'''
def build_bc_from_waveform_ac(
    wave_ac,
    rate=16000,
    folder_clean_air="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/vibravox_final/clean_air_train",
    folder_clean_bone="/home/ms_ablasioli/alessandra/denoiser_backup_bcn/clean_bone_train",
    T=5,
    seg_len_mic=640,
    overlap_mic=320,
    segment=0.83,
    stride=0.83
):
    """
    Converts an air conduction waveform into a bone conduction waveform
    using a response estimated from all clean files in the clean_air and clean_bone folders.
    
    Args:
        wave_ac: numpy array or torch tensor 1D, air conduction signal
        rate: sample rate
        folder_clean_air: folder containing clean air files
        folder_clean_bone: folder containing clean bone files
        T, seg_len_mic, overlap_mic, segment, stride: framing parameters

    Returns:
        bc_wave: torch tensor, estimated bone conduction waveform, shape [1, N]
    """

    # --- Get sorted list of clean files and check they match ---
    air_files = sorted(os.listdir(folder_clean_air))
    bone_files = sorted(os.listdir(folder_clean_bone))
    assert len(air_files) == len(bone_files), "Clean air and bone folders must have the same number of files"

    # --- Derived parameters for STFT framing ---
    freq_bin_high = int(rate / rate * int(seg_len_mic / 2)) + 1
    time_bin = int(segment * rate / (seg_len_mic - overlap_mic)) + 1
    time_stride = int(stride * rate / (seg_len_mic - overlap_mic))

    # --- Initialize response matrix ---
    response = np.zeros((2, freq_bin_high))

    # --- Loop over all clean file pairs to accumulate the response ---
    for f_air, f_bone in zip(air_files, bone_files):
        path_air = os.path.join(folder_clean_air, f_air)
        path_bone = os.path.join(folder_clean_bone, f_bone)

        # Load clean AC and BC files and compute their STFT
        _, Zxx_air_clean, _ = load_audio_n(path_air, T, seg_len_mic, overlap_mic, rate, normalize=True)
        _, Zxx_bone_clean, _ = load_audio_n(path_bone, T, seg_len_mic, overlap_mic, rate, normalize=True)

        # Update the response with clips from each segment
        for j in range(int((T - segment) / stride) + 1):
            clip2 = Zxx_air_clean[:freq_bin_high, j * time_stride:j * time_stride + time_bin]
            clip1 = Zxx_bone_clean[:, j * time_stride:j * time_stride + time_bin]
            response = transfer_function(clip1, clip2, response)

    # --- Average the accumulated response over all clean files ---
    response /= len(air_files)
    f = response[0]
    v = response[1]

    # --- Convert input AC waveform to numpy if it's a torch tensor ---
    if isinstance(wave_ac, torch.Tensor):
        wave_ac = wave_ac.numpy().squeeze()

    # Compute STFT of the input AC waveform
    Zxx_air, phase = frequencydomain(wave_ac, seg_len=seg_len_mic, overlap=overlap_mic, rate=rate, mfcc=False)

    # --- Apply the response to the AC spectrogram ---
    time_bin_test = Zxx_air.shape[-1]
    response_matrix = np.tile(np.expand_dims(f, axis=1), (1, time_bin_test))
    for j in range(time_bin_test):
        response_matrix[:, j] += np.random.normal(0, v, (freq_bin_high))

    acc = response_matrix / np.max(f) * Zxx_air[:, :]

    # --- Reconstruct the BC waveform from the modified spectrogram ---
    Zxx_rec = acc * phase
    _, bc_wave = signal.istft(Zxx_rec, nperseg=seg_len_mic, noverlap=overlap_mic, fs=rate)

    # Return as torch tensor with a channel dimension
    return torch.tensor(bc_wave).unsqueeze(0).float()

'''

def build_bc_from_waveform_ac(
    wave_ac,
    rate=16000,
    response_path="response.npy",
    seg_len_mic=640,
    overlap_mic=320
):
    """
    Converts an air conduction waveform into bone conduction waveform
    using a precomputed response (saved with compute_and_save_response).
    """
    # Load response
    response = np.load(response_path)
    f = response[0]
    v = response[1]

    # Convert torch tensor to numpy
    if isinstance(wave_ac, torch.Tensor):
        wave_ac = wave_ac.numpy().squeeze()

    # STFT of AC
    Zxx_air, phase = frequencydomain(wave_ac, seg_len=seg_len_mic,
                                     overlap=overlap_mic, rate=rate, mfcc=False)

    # Apply response
    freq_bin_high = len(f)
    time_bin_test = Zxx_air.shape[-1]
    response_matrix = np.tile(f[:, None], (1, time_bin_test))

    for j in range(time_bin_test):
        response_matrix[:, j] += np.random.normal(0, v, freq_bin_high)

    acc = response_matrix / np.max(f) * Zxx_air

    # Reconstruct BC waveform
    Zxx_rec = acc * phase
    _, bc_wave = signal.istft(Zxx_rec, nperseg=seg_len_mic,
                              noverlap=overlap_mic, fs=rate)

    return torch.tensor(bc_wave).unsqueeze(0).float()

