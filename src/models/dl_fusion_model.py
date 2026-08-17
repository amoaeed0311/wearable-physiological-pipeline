"""
src/models/dl_fusion_model.py
1D-CNN Deep Learning Architecture for End-to-End Multimodal Sensor Fusion in PyTorch.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.preprocessing import synchronize_and_filter_subject, FS_PPG
from src.ingestion.load_data import download_sample_data


# 1. PyTorch Dataset for Windowed Multimodal Signals
class WearableSensorDataset(Dataset):
    def __init__(self, raw_matrix, labels, window_sec=8, step_sec=2, fs=64):
        self.window_samples = int(window_sec * fs)
        self.step_samples = int(step_sec * fs)
        
        # Channels 0-3: [Filtered_PPG, ACC_X, ACC_Y, ACC_Z]
        self.data = raw_matrix[:, 0:4]
        self.labels = labels
        self.num_windows = (self.data.shape[0] - self.window_samples) // self.step_samples
        self.valid_len = min(self.num_windows, len(self.labels))

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        start = idx * self.step_samples
        end = start + self.window_samples
        # Shape: (Channels, Sequence_Length) = (4, 512)
        x = self.data[start:end, :].T
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 2. 1D-CNN Fusion Neural Network
class Multimodal1DCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.regressor(features)
        return output.squeeze(-1)


# 3. Training & Evaluation Pipeline
def train_and_eval_dl():
    raw_path = download_sample_data()
    sync_matrix, ground_truth_hr = synchronize_and_filter_subject(raw_path)

    full_dataset = WearableSensorDataset(sync_matrix, ground_truth_hr)
    split_idx = int(len(full_dataset) * 0.8)
    
    train_set = torch.utils.data.Subset(full_dataset, range(0, split_idx))
    test_set = torch.utils.data.Subset(full_dataset, range(split_idx, len(full_dataset)))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = Multimodal1DCNN(in_channels=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    print("[*] Training PyTorch 1D-CNN Multimodal Model (15 Epochs)...")
    model.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1:02d}/15] - Loss: {epoch_loss / len(train_set):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            all_preds.extend(preds.numpy())
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print("\n--- ⚡ PyTorch 1D-CNN Multimodal Model Performance ---")
    print(f"Test MAE : {mae:.3f} BPM")
    print(f"Test RMSE: {rmse:.3f} BPM")

    # Save model weights
    save_path = os.path.join(PROJECT_ROOT, "data", "processed", "multimodal_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Model weights saved to {save_path}")

    return model, mae, rmse


if __name__ == "__main__":
    train_and_eval_dl()