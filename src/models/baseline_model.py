"""
src/models/baseline_model.py
Classical Machine Learning Baseline (Random Forest on Handcrafted Features).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def train_baseline():
    csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "hrv_feature_matrix.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature matrix not found at {csv_path}. Run Phase 1 first.")

    df = pd.read_csv(csv_path)

    feature_cols = [
        'hr_estimated_bpm', 'sdnn_ms', 'rmssd_ms', 'pnn50_pct',
        'lf_power', 'hf_power', 'lf_hf_ratio',
        'motion_mean', 'motion_std', 'temp_mean'
    ]
    target_col = 'target_hr_bpm'

    # Chronological time-series split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- 🌲 Classical Random Forest Baseline Performance ---")
    print(f"Test MAE : {mae:.3f} BPM")
    print(f"Test RMSE: {rmse:.3f} BPM")

    # Feature Importance analysis
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 3 Important Features:")
    for feat, imp in importances.head(3).items():
        print(f"  - {feat}: {imp:.4f}")

    return model, mae, rmse


if __name__ == "__main__":
    train_baseline()