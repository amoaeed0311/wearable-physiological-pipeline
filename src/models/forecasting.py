"""
src/models/forecasting.py
Short-horizon physiological trend forecasting using PyTorch LSTM vs. Autoregressive baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def create_forecasting_sequences(series, lookback=15, horizon=5):
    """
    Transforms 1D time-series into overlapping (lookback, horizon) sliding window pairs.
    """
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class PhysiologicalLSTMForecaster(nn.Module):
    """
    LSTM-based sequence forecaster for biometric trends.
    """
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        _, (hn, _) = self.lstm(x)
        # hn[-1] shape: (batch_size, hidden_dim)
        out = self.fc(hn[-1])
        return out


def run_forecasting_pipeline():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "hrv_feature_matrix.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Feature matrix not found. Run Phase 1 first.")

    df = pd.read_csv(csv_path)
    hr_series = df['target_hr_bpm'].values

    LOOKBACK = 15   # Past 30 seconds of context (2-second step size)
    HORIZON = 5     # Next 10 seconds forecast

    X, y = create_forecasting_sequences(hr_series, lookback=LOOKBACK, horizon=HORIZON)

    # Chronological Split (80% Train, 20% Test)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # 1. Baseline: Persistence (Last-Known Value) Forecast
    y_pred_naive = np.repeat(X_test[:, -1:], HORIZON, axis=1)
    naive_mae = mean_absolute_error(y_test, y_pred_naive)
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))

    print("\n--- ⏳ Baseline: Persistence Forecaster ---")
    print(f"Test MAE (Next {HORIZON*2}s) : {naive_mae:.3f} BPM")
    print(f"Test RMSE (Next {HORIZON*2}s): {naive_rmse:.3f} BPM")

    # 2. PyTorch LSTM Forecaster
    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test).unsqueeze(-1)

    model = PhysiologicalLSTMForecaster(input_dim=1, hidden_dim=32, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("\n[*] Training LSTM Physiological Forecaster...")
    model.train()
    for epoch in range(40):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        lstm_preds = model(X_test_t).numpy()

    lstm_mae = mean_absolute_error(y_test, lstm_preds)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))

    print("\n--- ⚡ PyTorch LSTM Biometric Trend Forecaster ---")
    print(f"Test MAE (Next {HORIZON*2}s) : {lstm_mae:.3f} BPM")
    print(f"Test RMSE (Next {HORIZON*2}s): {lstm_rmse:.3f} BPM")

    save_path = os.path.join(PROJECT_ROOT, "data", "processed", "lstm_forecaster.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[✓] Forecaster weights saved to {save_path}")

    return naive_mae, lstm_mae


if __name__ == "__main__":
    run_forecasting_pipeline()