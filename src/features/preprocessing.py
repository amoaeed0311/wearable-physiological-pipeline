"""
src/features/preprocessing.py
Signal processing, filtering, and multimodal synchronization.
"""

import os
import sys
import pickle
import numpy as np
from scipy import signal

# Add project root to sys.path automatically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FS_PPG = 64
FS_ACC = 32
FS_TEMP = 4


def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=64, order=3):
    """
    Zero-phase Butterworth bandpass filter to isolate cardiac frequencies (30-240 BPM).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered


def synchronize_and_filter_subject(file_path):
    """
    Loads raw subject stream, cleans PPG with bandpass filtering,
    resamples all channels to uniform 64 Hz, and outputs time-locked tabular windows.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    raw_ppg = data['signal']['wrist']['BVP'].flatten()
    raw_acc = data['signal']['wrist']['ACC']
    raw_temp = data['signal']['wrist']['TEMP'].flatten()
    labels = data['label']

    # 1. Bandpass filter PPG
    filtered_ppg = butter_bandpass_filter(raw_ppg, lowcut=0.5, highcut=4.0, fs=FS_PPG)

    # 2. Resample ACC and TEMP to match PPG (64 Hz)
    num_samples = len(raw_ppg)
    time_target = np.linspace(0, num_samples / FS_PPG, num_samples)
    
    time_acc = np.linspace(0, len(raw_acc) / FS_ACC, len(raw_acc))
    acc_x = np.interp(time_target, time_acc, raw_acc[:, 0])
    acc_y = np.interp(time_target, time_acc, raw_acc[:, 1])
    acc_z = np.interp(time_target, time_acc, raw_acc[:, 2])
    
    # 3. Calculate motion intensity norm
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    time_temp = np.linspace(0, len(raw_temp) / FS_TEMP, len(raw_temp))
    resampled_temp = np.interp(time_target, time_temp, raw_temp)

    # 4. Construct synchronized multi-channel matrix
    # Channels: [Filtered_PPG, ACC_X, ACC_Y, ACC_Z, ACC_Magnitude, Temperature]
    synchronized_features = np.column_stack([
        filtered_ppg,
        acc_x,
        acc_y,
        acc_z,
        acc_mag,
        resampled_temp
    ])

    print(f"[✓] Multi-sensor stream synchronized. Matrix shape: {synchronized_features.shape}")
    return synchronized_features, labels


if __name__ == "__main__":
    from src.ingestion.load_data import download_sample_data
    path = download_sample_data()
    features, labels = synchronize_and_filter_subject(path)