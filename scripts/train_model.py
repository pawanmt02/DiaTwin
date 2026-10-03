"""
DiaTwin Training Script

End-to-end pipeline: load data → preprocess → create datasets → train → evaluate.
Run from project root: python scripts/train_model.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from backend.app.core.config import config as cfg
from backend.app.core.preprocessing import (
    StaticEHRPreprocessor,
    DynamicCGMPreprocessor,
    DiaTwinDataset,
    create_dataloaders,
)
from backend.app.models.forecaster import build_model, count_parameters
from backend.app.models.trainer import Trainer
from backend.app.services.evaluation import full_evaluation


def main():
    print("=" * 60)
    print("  DiaTwin Model Training Pipeline")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load EHR profiles
    # ------------------------------------------------------------------
    print("\n[1/6] Loading EHR profiles...")
    ehr_df = pd.read_csv(cfg.data.ehr_profiles_path)
    print(f"  Loaded {len(ehr_df)} patient profiles")

    # ------------------------------------------------------------------
    # 2. Preprocess static EHR
    # ------------------------------------------------------------------
    print("\n[2/6] Preprocessing static EHR features...")
    ehr_preprocessor = StaticEHRPreprocessor()
    static_matrix = ehr_preprocessor.fit_transform(ehr_df)
    static_dim = static_matrix.shape[1]
    print(f"  Static feature dimension: {static_dim}")

    # Map patient_id to static vector
    patient_ids = ehr_df["patient_id"].tolist()
    patient_static = {
        pid: static_matrix[i] for i, pid in enumerate(patient_ids)
    }

    # Save preprocessor
    ehr_preprocessor.save(os.path.join(cfg.data.processed_data_dir, "ehr_preprocessor.pkl"))

    # ------------------------------------------------------------------
    # 3. Load and preprocess CGM data
    # ------------------------------------------------------------------
    print("\n[3/6] Loading and engineering CGM features...")
    cgm_preprocessor = DynamicCGMPreprocessor()
    feature_cols = cgm_preprocessor.get_feature_columns()
    dynamic_dim = len(feature_cols)

    patient_dynamic_data = {}
    patient_glucose_raw = {}
    all_features = []

    cgm_dir = cfg.data.cgm_data_dir
    loaded = 0

    # Check if CGM is stored as individual files or single file
    if os.path.isdir(cgm_dir):
        cgm_files = sorted([f for f in os.listdir(cgm_dir) if f.endswith(".csv")])
    else:
        cgm_files = []

    # Limit patients for training speed (use subset)
    max_patients = min(50, len(cgm_files))  # Use 50 patients for CPU training
    print(f"  Processing {max_patients} patient CGM files...")

    for fname in cgm_files[:max_patients]:
        pid = fname.replace(".csv", "")
        if pid not in patient_static:
            continue

        fpath = os.path.join(cgm_dir, fname)
        try:
            df = pd.read_csv(fpath)
            df = cgm_preprocessor.engineer_features(df)

            features = df[feature_cols].values.astype(np.float32)
            glucose_raw = df["glucose_mgdl"].values.astype(np.float32)

            # Skip if too short
            if len(features) < cfg.model.lookback_window + cfg.model.forecast_horizon + 10:
                continue

            all_features.append(features)
            patient_dynamic_data[pid] = features
            patient_glucose_raw[pid] = glucose_raw
            loaded += 1

        except Exception as e:
            print(f"  Warning: Failed to process {fname}: {e}")

    if loaded == 0:
        print("  No CGM data found. Generating synthetic sequences...")
        # Generate synthetic data for demo/training
        np.random.seed(42)
        for pid in patient_ids[:100]:
            T = 2016  # 7 days of 5-min data
            base = float(ehr_df[ehr_df["patient_id"] == pid]["fasting_glucose_mgdl"].iloc[0])

            # Realistic glucose pattern
            t = np.arange(T)
            circadian = 15 * np.sin(2 * np.pi * t / 288)  # Daily cycle
            meals = np.zeros(T)
            for day in range(7):
                for meal_time in [96, 156, 240]:  # 8am, 1pm, 8pm
                    idx = day * 288 + meal_time
                    if idx < T:
                        meal_response = 50 * np.exp(-np.arange(min(60, T - idx)) / 15)
                        meals[idx:idx + len(meal_response)] += meal_response

            noise = np.random.randn(T) * 3
            glucose = base + circadian + meals + noise
            glucose = np.clip(glucose, 40, 400).astype(np.float32)

            # Build features
            delta = np.diff(glucose, prepend=glucose[0])
            delta2 = np.diff(delta, prepend=delta[0])
            steps = np.random.poisson(30, T).astype(np.float32)
            steps_15 = np.convolve(steps, np.ones(3), mode="same")
            steps_30 = np.convolve(steps, np.ones(6), mode="same")
            steps_60 = np.convolve(steps, np.ones(12), mode="same")
            hr = np.random.normal(75, 5, T).astype(np.float32)
            t_min = (t * 5) % 1440
            t_sin = np.sin(2 * np.pi * t_min / 1440).astype(np.float32)
            t_cos = np.cos(2 * np.pi * t_min / 1440).astype(np.float32)

            features = np.column_stack([
                glucose, delta, delta2, steps_15, steps_30, steps_60,
                hr, t_sin, t_cos
            ]).astype(np.float32)

            all_features.append(features)
            patient_dynamic_data[pid] = features
            patient_glucose_raw[pid] = glucose
            loaded += 1

    print(f"  Loaded {loaded} patients with CGM data")

    # Fit dynamic scaler
    all_features_concat = np.vstack(all_features)
    cgm_preprocessor.fit(all_features_concat)

    # Scale dynamic features
    for pid in patient_dynamic_data:
        patient_dynamic_data[pid] = cgm_preprocessor.transform(patient_dynamic_data[pid])

    # Save preprocessor
    cgm_preprocessor.save(os.path.join(cfg.data.processed_data_dir, "cgm_preprocessor.pkl"))

    # ------------------------------------------------------------------
    # 4. Create Dataset & DataLoaders
    # ------------------------------------------------------------------
    print("\n[4/6] Creating PyTorch dataset...")
    dataset = DiaTwinDataset(
        patient_dynamic_data=patient_dynamic_data,
        patient_static_data=patient_static,
        patient_glucose_raw=patient_glucose_raw,
        lookback=cfg.model.lookback_window,
        horizon=cfg.model.forecast_horizon,
    )
    print(f"  Total samples: {len(dataset):,}")

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=cfg.model.batch_size,
        train_split=0.8,
        val_split=0.1,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}")

    # ------------------------------------------------------------------
    # 5. Build & Train Model
    # ------------------------------------------------------------------
    print("\n[5/6] Building and training model...")
    model = build_model(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        device=device,
    )
    print(f"  Model parameters: {count_parameters(model):,}")

    trainer = Trainer(model, device=device)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.model.num_epochs,
        patience=cfg.model.early_stopping_patience,
    )

    # Save training history
    os.makedirs(cfg.data.processed_data_dir, exist_ok=True)
    with open(os.path.join(cfg.data.processed_data_dir, "training_history.json"), "w") as f:
        json.dump({
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "best_val_loss": history["best_val_loss"],
            "epochs_trained": history["epochs_trained"],
        }, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Evaluate on Test Set
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating on test set...")
    model.eval()

    all_true = []
    all_pred = []
    all_true_seq = []
    all_pred_seq = []

    with torch.no_grad():
        for static, dynamic, target in test_loader:
            static = static.to(device)
            dynamic = dynamic.to(device)

            preds = model(static, dynamic)  # [B, H, 3]
            median_preds = preds[:, :, 1].cpu().numpy()  # 50th percentile

            all_true.append(target.numpy())
            all_pred.append(median_preds)
            all_true_seq.append(target.numpy())
            all_pred_seq.append(median_preds)

    y_true = np.concatenate(all_true).flatten()
    y_pred = np.concatenate(all_pred).flatten()
    y_true_seq = np.concatenate(all_true_seq)
    y_pred_seq = np.concatenate(all_pred_seq)

    eval_dir = os.path.join(cfg.data.processed_data_dir, "evaluation")
    report = full_evaluation(y_true, y_pred, y_true_seq, y_pred_seq, save_dir=eval_dir)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MARD:  {report.mard:.2f}%  (target: <10%)")
    print(f"  RMSE:  {report.rmse:.2f} mg/dL  (target: <18 mg/dL)")
    print(f"  MAE:   {report.mae:.2f} mg/dL")
    print(f"  Clarke EGA Zones:")
    for zone, pct in report.clarke_zones.items():
        print(f"    Zone {zone}: {pct:.1f}%")
    ab_pct = report.clarke_zones.get("A", 0) + report.clarke_zones.get("B", 0)
    print(f"    A+B Combined: {ab_pct:.1f}%  (target: >98%)")
    print(f"  Hypo Sensitivity:  {report.hypo_sensitivity:.2%}")
    print(f"  Hyper Sensitivity: {report.hyper_sensitivity:.2%}")
    print(f"  Mean Alert Lead Time: {report.mean_lead_time_min:.0f} min  (target: ≥45 min)")
    print("=" * 60)

    # Save evaluation report
    eval_report = {
        "mard": report.mard,
        "rmse": report.rmse,
        "mae": report.mae,
        "clarke_zones": report.clarke_zones,
        "clarke_ab_combined": ab_pct,
        "hypo_sensitivity": report.hypo_sensitivity,
        "hyper_sensitivity": report.hyper_sensitivity,
        "mean_lead_time_min": report.mean_lead_time_min,
    }
    with open(os.path.join(eval_dir, "evaluation_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nResults saved to {eval_dir}/")
    print("Training complete! ✓")


if __name__ == "__main__":
    main()
