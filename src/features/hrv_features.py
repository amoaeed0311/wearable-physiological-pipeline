"""
src/features/hrv_features.py
Feature extraction pipeline: HRV time/frequency metrics and motion artifact indices.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal, integrate

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.preprocessing import synchronize_and_filter_subject, FS_PPG
from src.ingestion.load_data import download_sample_data


def extract_window_features(ppg_window, acc_mag_window, temp_window, fs=64):
    """
    Extracts physiological time-domain, frequency-domain, and motion features from a single window.
    """
    # 1. Peak detection on bandpassed PPG to obtain inter-beat intervals (IBI)
    min_distance = int(fs * 0.3)  # Max 200 BPM
    peaks, _ = signal.find_peaks(ppg_window, distance=min_distance, prominence=0.2)
    
    if len(peaks) > 2:
        ibi_ms = np.diff(peaks) / fs * 1000.0
        hr_bpm = 60000.0 / np.mean(ibi_ms)
        sdnn = np.std(ibi_ms)
        rmssd = np.sqrt(np.mean(np.square(np.diff(ibi_ms)))) if len(ibi_ms) > 1 else 0.0
        pnn50 = np.sum(np.abs(np.diff(ibi_ms)) > 50) / len(ibi_ms) * 100 if len(ibi_ms) > 1 else 0.0
    else:
        hr_bpm = 70.0
        sdnn = 0.0
        rmssd = 0.0
        pnn50 = 0.0

    # 2. Frequency-domain power spectral density via Welch's method
    freqs, psd = signal.welch(ppg_window, fs=fs, nperseg=min(len(ppg_window), 256))
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.40)
    
    # Use scipy.integrate.trapezoid for universal compatibility
    lf_power = float(integrate.trapezoid(psd[lf_band], freqs[lf_band])) if np.any(lf_band) else 1e-6
    hf_power = float(integrate.trapezoid(psd[hf_band], freqs[hf_band])) if np.any(hf_band) else 1e-6
    lf_hf_ratio = lf_power / (hf_power + 1e-6)

    # 3. Motion & Thermal statistics
    motion_mean = float(np.mean(acc_mag_window))
    motion_std = float(np.std(acc_mag_window))
    temp_mean = float(np.mean(temp_window))

    return {
        'hr_estimated_bpm': hr_bpm,
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'pnn50_pct': pnn50,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_hf_ratio,
        'motion_mean': motion_mean,
        'motion_std': motion_std,
        'temp_mean': temp_mean
    }


def generate_sliding_window_dataset(window_sec=8, step_sec=2):
    """
    Constructs a tabular dataset of physiological features matching ground-truth HR labels.
    """
    raw_path = download_sample_data()
    sync_matrix, ground_truth_hr = synchronize_and_filter_subject(raw_path)
    
    window_samples = int(window_sec * FS_PPG)
    step_samples = int(step_sec * FS_PPG)
    
    rows = []
    num_windows = (sync_matrix.shape[0] - window_samples) // step_samples

    for i in range(num_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        
        ppg_w = sync_matrix[start_idx:end_idx, 0]
        acc_mag_w = sync_matrix[start_idx:end_idx, 4]
        temp_w = sync_matrix[start_idx:end_idx, 5]
        
        feats = extract_window_features(ppg_w, acc_mag_w, temp_w, fs=FS_PPG)
        
        # Match label
        if i < len(ground_truth_hr):
            feats['target_hr_bpm'] = ground_truth_hr[i]
            rows.append(feats)

    df_features = pd.DataFrame(rows)
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "hrv_feature_matrix.csv")
    df_features.to_csv(processed_path, index=False)
    print(f"[✓] Feature engineering complete! Saved {df_features.shape[0]} windows to {processed_path}")
    print("\n--- Preview of Engineered Physiological Features ---")
    print(df_features[['hr_estimated_bpm', 'sdnn_ms', 'rmssd_ms', 'lf_hf_ratio', 'motion_mean', 'target_hr_bpm']].head())
    return df_features


if __name__ == "__main__":
    generate_sliding_window_dataset()