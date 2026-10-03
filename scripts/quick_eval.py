"""
DiaTwin Quick Evaluation Script

Loads the saved checkpoint, runs evaluation on a small test set,
and generates the Clarke Error Grid plot + metrics report.
Run: python scripts/quick_eval.py
"""

import os
import sys
import json
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.app.core.config import config as cfg
from backend.app.models.forecaster import build_model, count_parameters
from backend.app.services.evaluation import full_evaluation


def main():
    print("=" * 60)
    print("  DiaTwin Quick Evaluation")
    print("=" * 60)

    device = "cpu"
    model_path = cfg.model.best_model_path

    if not os.path.exists(model_path):
        print("No model checkpoint found. Run training first:")
        print("  python scripts/train_model.py")
        return

    # Load model
    print("Loading model...")
    model = build_model(static_dim=23, dynamic_dim=9, device=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Parameters: {count_parameters(model):,}")

    # Generate synthetic test data for evaluation
    print("Generating test predictions...")
    np.random.seed(123)
    N = 500  # test samples
    all_true = []
    all_pred = []

    for _ in range(N):
        # Random static vector
        x_static = torch.randn(1, 23)
        # Random glucose-like dynamic sequence
        base = np.random.uniform(90, 200)
        glucose = base + np.cumsum(np.random.randn(24) * 3)
        glucose = np.clip(glucose, 50, 350)

        delta = np.diff(glucose, prepend=glucose[0])
        delta2 = np.diff(delta, prepend=delta[0])
        steps = np.random.poisson(30, 24).astype(float)
        hr = np.random.normal(75, 5, 24)
        t = np.arange(24)
        t_sin = np.sin(2 * np.pi * (t * 5 % 1440) / 1440)
        t_cos = np.cos(2 * np.pi * (t * 5 % 1440) / 1440)

        dynamic = np.column_stack([
            glucose, delta, delta2,
            steps, steps * 2, steps * 4,
            hr, t_sin, t_cos
        ]).astype(np.float32)

        x_dynamic = torch.tensor(dynamic).unsqueeze(0)

        with torch.no_grad():
            preds = model(x_static, x_dynamic)  # [1, 24, 3]

        # Use median prediction
        pred_median = preds[0, :, 1].numpy()

        # Simulate "true" future as continuation with noise
        true_future = base + np.cumsum(np.random.randn(24) * 4)
        true_future = np.clip(true_future, 50, 350)

        all_true.append(true_future)
        all_pred.append(pred_median)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_true_seq = np.array(all_true)
    y_pred_seq = np.array(all_pred)

    # Run evaluation
    eval_dir = os.path.join(cfg.data.processed_data_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    report = full_evaluation(y_true, y_pred, y_true_seq, y_pred_seq, save_dir=eval_dir)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MARD:  {report.mard:.2f}%")
    print(f"  RMSE:  {report.rmse:.2f} mg/dL")
    print(f"  MAE:   {report.mae:.2f} mg/dL")
    print(f"  Clarke EGA Zones:")
    for zone, pct in report.clarke_zones.items():
        print(f"    Zone {zone}: {pct:.1f}%")
    ab_pct = report.clarke_zones.get("A", 0) + report.clarke_zones.get("B", 0)
    print(f"    A+B Combined: {ab_pct:.1f}%")
    print(f"  Hypo Sensitivity:  {report.hypo_sensitivity:.2%}")
    print(f"  Hyper Sensitivity: {report.hyper_sensitivity:.2%}")
    print(f"  Mean Alert Lead Time: {report.mean_lead_time_min:.0f} min")
    print("=" * 60)

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

    print(f"\nClarke Error Grid plot saved to {eval_dir}/clarke_error_grid.png")
    print(f"Report saved to {eval_dir}/evaluation_report.json")
    print("Done!")


if __name__ == "__main__":
    main()
