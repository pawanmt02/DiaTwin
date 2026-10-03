"""
DiaTwin Data Preprocessing Pipeline

Handles both static EHR preprocessing (normalization, encoding) and
dynamic CGM time-series preprocessing (interpolation, feature engineering,
temporal encoding, windowed dataset creation).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import RobustScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import joblib

from backend.app.core.config import config


# ---------------------------------------------------------------------------
# Static EHR Preprocessor
# ---------------------------------------------------------------------------

class StaticEHRPreprocessor:
    """Preprocesses static Electronic Health Record data.

    Applies RobustScaler normalization (resistant to outliers common in
    clinical data) and label encoding for categorical variables.
    """

    def __init__(self):
        self.numerical_scaler = RobustScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "StaticEHRPreprocessor":
        """Fit scalers and encoders on training EHR data."""
        num_cols = [c for c in config.data.static_numerical_features if c in df.columns]
        cat_cols = [c for c in config.data.static_categorical_features if c in df.columns]

        # Fit numerical scaler
        self.numerical_scaler.fit(df[num_cols].fillna(df[num_cols].median()))

        # Fit label encoders for categorical columns
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna("unknown"))
            self.label_encoders[col] = le

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform EHR dataframe into numerical feature vector."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")

        # Numerical features: impute with median then scale
        num_data = df[self.num_cols].fillna(df[self.num_cols].median())
        num_scaled = self.numerical_scaler.transform(num_data)

        # Categorical features: encode to integers
        cat_encoded = np.zeros((len(df), len(self.cat_cols)), dtype=np.float32)
        for i, col in enumerate(self.cat_cols):
            encoded = df[col].astype(str).fillna("unknown")
            # Handle unseen categories gracefully
            known = set(self.label_encoders[col].classes_)
            encoded = encoded.apply(lambda x: x if x in known else "unknown")
            cat_encoded[:, i] = self.label_encoders[col].transform(encoded)

        return np.hstack([num_scaled.astype(np.float32), cat_encoded])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, path: str):
        """Persist preprocessor state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "scaler": self.numerical_scaler,
            "encoders": self.label_encoders,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
        }, path)

    def load(self, path: str) -> "StaticEHRPreprocessor":
        """Load preprocessor state."""
        state = joblib.load(path)
        self.numerical_scaler = state["scaler"]
        self.label_encoders = state["encoders"]
        self.num_cols = state["num_cols"]
        self.cat_cols = state["cat_cols"]
        self.is_fitted = True
        return self


# ---------------------------------------------------------------------------
# Dynamic CGM Time-Series Preprocessor
# ---------------------------------------------------------------------------

class DynamicCGMPreprocessor:
    """Preprocesses continuous glucose monitoring time-series data.

    Performs: linear interpolation for gaps, rolling statistics,
    rate-of-change features, and circadian temporal encoding.
    """

    def __init__(self):
        self.glucose_scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw CGM stream.

        Args:
            df: DataFrame with columns [timestamp, glucose_mgdl, steps_5min, heart_rate_bpm]

        Returns:
            DataFrame with engineered features added.
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by time
        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- Glucose derivatives ---
        # Rate of change (first difference)
        df["delta_glucose"] = df["glucose_mgdl"].diff().fillna(0)
        # Acceleration (second difference)
        df["delta2_glucose"] = df["delta_glucose"].diff().fillna(0)

        # --- Rolling step counts ---
        # 15 min = 3 readings, 30 min = 6 readings, 60 min = 12 readings
        df["steps_15min"] = df["steps_5min"].rolling(window=3, min_periods=1).sum()
        df["steps_30min"] = df["steps_5min"].rolling(window=6, min_periods=1).sum()
        df["steps_60min"] = df["steps_5min"].rolling(window=12, min_periods=1).sum()

        # --- Circadian temporal encoding ---
        minutes_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df["time_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
        df["time_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)

        # --- Linear interpolation for any gaps ---
        for col in ["glucose_mgdl", "heart_rate_bpm", "steps_5min"]:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

        return df

    def get_feature_columns(self) -> List[str]:
        """Return ordered list of dynamic feature columns."""
        return [
            "glucose_mgdl", "delta_glucose", "delta2_glucose",
            "steps_15min", "steps_30min", "steps_60min",
            "heart_rate_bpm", "time_sin", "time_cos"
        ]

    def fit(self, feature_matrix: np.ndarray) -> "DynamicCGMPreprocessor":
        """Fit scaler on training feature matrix."""
        self.feature_scaler.fit(feature_matrix)
        self.is_fitted = True
        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Scale features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")
        return self.feature_scaler.transform(feature_matrix).astype(np.float32)

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(feature_matrix).transform(feature_matrix)

    def save(self, path: str):
        """Persist scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"feature_scaler": self.feature_scaler}, path)

    def load(self, path: str) -> "DynamicCGMPreprocessor":
        """Load scaler."""
        state = joblib.load(path)
        self.feature_scaler = state["feature_scaler"]
        self.is_fitted = True
        return self


# ---------------------------------------------------------------------------
# PyTorch Dataset for Sliding Window Sequences
# ---------------------------------------------------------------------------

class DiaTwinDataset(Dataset):
    """PyTorch Dataset that creates sliding-window samples for the forecasting model.

    Each sample contains:
        - static_features: [static_dim] patient phenotype vector
        - dynamic_input: [lookback, dynamic_dim] past CGM window
        - target: [horizon] future glucose values (raw mg/dL for loss computation)
    """

    def __init__(
        self,
        patient_dynamic_data: Dict[str, np.ndarray],
        patient_static_data: Dict[str, np.ndarray],
        patient_glucose_raw: Dict[str, np.ndarray],
        lookback: int = 24,
        horizon: int = 24,
    ):
        """
        Args:
            patient_dynamic_data: {patient_id: [T, dynamic_features] scaled}
            patient_static_data: {patient_id: [static_dim] scaled}
            patient_glucose_raw: {patient_id: [T] raw glucose in mg/dL}
            lookback: Number of past time steps (L)
            horizon: Number of future time steps (H)
        """
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []

        for pid in patient_dynamic_data:
            dynamic = patient_dynamic_data[pid]       # [T, D]
            static = patient_static_data[pid]         # [S]
            glucose_raw = patient_glucose_raw[pid]     # [T]

            T = len(dynamic)
            # Create sliding windows
            for t in range(lookback, T - horizon):
                self.samples.append({
                    "static": torch.tensor(static, dtype=torch.float32),
                    "dynamic_input": torch.tensor(
                        dynamic[t - lookback: t], dtype=torch.float32
                    ),
                    "target": torch.tensor(
                        glucose_raw[t: t + horizon], dtype=torch.float32
                    ),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return sample["static"], sample["dynamic_input"], sample["target"]


def create_dataloaders(
    dataset: DiaTwinDataset,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split dataset and create train/val/test DataLoaders."""
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
