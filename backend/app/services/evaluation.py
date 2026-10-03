"""
DiaTwin Clinical Evaluation Module

Implements domain-standard bio-mathematical validation metrics:
  - Mean Absolute Relative Difference (MARD)
  - Root Mean Squared Error (RMSE)
  - Clarke Error Grid Analysis (EGA)
  - Adverse event detection metrics (sensitivity, specificity, lead time)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from backend.app.core.config import config as cfg


@dataclass
class EvaluationReport:
    """Container for all clinical evaluation metrics."""
    mard: float                     # Mean Absolute Relative Difference (%)
    rmse: float                     # Root Mean Squared Error (mg/dL)
    mae: float                      # Mean Absolute Error (mg/dL)
    clarke_zones: Dict[str, float]  # Percentage in each Clarke zone (A-E)
    hypo_sensitivity: float         # Sensitivity for hypoglycemia detection
    hypo_specificity: float         # Specificity for hypoglycemia detection
    hyper_sensitivity: float        # Sensitivity for hyperglycemia detection
    hyper_specificity: float        # Specificity for hyperglycemia detection
    mean_lead_time_min: float       # Average alert lead time (minutes)


def compute_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Relative Difference.

    MARD = (1/N) × Σ |y_true - y_pred| / y_true × 100%

    Standard clinical accuracy metric for CGM and prediction systems.
    Excludes near-zero reference values to avoid division issues.
    """
    mask = y_true > 20  # Exclude unrealistic low values
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    )


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in mg/dL."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in mg/dL."""
    return float(np.mean(np.abs(y_true - y_pred)))


def clarke_error_zone(ref: float, pred: float) -> str:
    """Determine Clarke Error Grid zone for a single (reference, prediction) pair.

    Zones:
        A: Clinically accurate
        B: Benign errors (would not lead to inappropriate treatment)
        C: Unnecessary treatment
        D: Dangerous failure to detect
        E: Erroneous treatment (opposite direction)
    """
    if ref <= 70 and pred <= 70:
        return "A"
    if ref >= 180 and pred >= 180:
        return "A"
    if 70 <= ref <= 180:
        if abs(pred - ref) <= 20 or abs(pred - ref) / ref <= 0.20:
            return "A"

    if ref >= 180 and pred >= ref * 0.70:
        return "B"
    if ref <= 70 and pred <= ref * 1.30 + 20:
        return "B"

    # Check for D zone (dangerous failure to detect)
    if ref > 240 and pred < 70:
        return "E"
    if ref < 70 and pred > 180:
        return "E"

    if ref > 180 and pred < 70:
        return "D"
    if ref < 70 and pred > 180:
        return "D"

    # Remaining are C or B
    if ref >= 70 and pred < 70:
        return "C"
    if ref <= 180 and pred > 180:
        return "C"

    return "B"


def clarke_error_grid_analysis(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute Clarke Error Grid zone percentages.

    Returns dict with keys A, B, C, D, E and percentage values.
    """
    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    n = len(y_true)

    for i in range(n):
        zone = clarke_error_zone(float(y_true[i]), float(y_pred[i]))
        zones[zone] += 1

    # Convert to percentages
    return {k: round(v / n * 100, 2) for k, v in zones.items()}


def plot_clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Clarke Error Grid Analysis",
) -> plt.Figure:
    """Generate Clarke Error Grid plot.

    Plots reference vs. predicted glucose with zone boundaries.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Zone boundary lines
    ax.plot([0, 400], [0, 400], "k-", lw=1)  # Perfect agreement line

    # Zone A boundaries
    ax.plot([0, 175 / 3], [70, 70], "k-", lw=0.5)
    ax.plot([175 / 3, 400 / 1.2], [70, 400], "k-", lw=0.5)
    ax.plot([70, 70], [0, 56], "k-", lw=0.5)
    ax.plot([70, 400], [56, 400 * 0.8], "k-", lw=0.5)

    # Threshold lines
    ax.axhline(y=70, color="gray", linestyle="--", alpha=0.5, label="70 mg/dL")
    ax.axhline(y=180, color="gray", linestyle="--", alpha=0.5, label="180 mg/dL")
    ax.axvline(x=70, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=180, color="gray", linestyle="--", alpha=0.5)

    # Color points by zone
    zones_dict = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for i in range(len(y_true)):
        zone = clarke_error_zone(float(y_true[i]), float(y_pred[i]))
        zones_dict[zone].append((y_true[i], y_pred[i]))

    colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c", "E": "#8e44ad"}
    for zone, points in zones_dict.items():
        if points:
            refs, preds = zip(*points)
            ax.scatter(refs, preds, c=colors[zone], s=8, alpha=0.5, label=f"Zone {zone}")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compute_adverse_event_metrics(
    y_true_sequences: np.ndarray,
    y_pred_sequences: np.ndarray,
    threshold_low: float = None,
    threshold_high: float = None,
    sampling_interval_min: int = 5,
) -> Dict[str, float]:
    """Compute sensitivity/specificity for adverse event detection.

    Evaluates how well the model detects upcoming hypo/hyperglycemic events
    and the average lead time of warnings.

    Args:
        y_true_sequences: [N, horizon] true future glucose
        y_pred_sequences: [N, horizon] predicted future glucose (median quantile)
        threshold_low: Hypoglycemia threshold (default: 70 mg/dL)
        threshold_high: Hyperglycemia threshold (default: 180 mg/dL)
        sampling_interval_min: Minutes between readings
    """
    threshold_low = threshold_low or cfg.clinical.hypoglycemia_threshold
    threshold_high = threshold_high or cfg.clinical.hyperglycemia_threshold

    # Binary event detection
    true_hypo = np.any(y_true_sequences < threshold_low, axis=1)
    pred_hypo = np.any(y_pred_sequences < threshold_low, axis=1)
    true_hyper = np.any(y_true_sequences > threshold_high, axis=1)
    pred_hyper = np.any(y_pred_sequences > threshold_high, axis=1)

    def _sens_spec(true_events, pred_events):
        tp = np.sum(true_events & pred_events)
        tn = np.sum(~true_events & ~pred_events)
        fp = np.sum(~true_events & pred_events)
        fn = np.sum(true_events & ~pred_events)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        return float(sensitivity), float(specificity)

    hypo_sens, hypo_spec = _sens_spec(true_hypo, pred_hypo)
    hyper_sens, hyper_spec = _sens_spec(true_hyper, pred_hyper)

    # Lead time: how many steps before the actual event does the prediction fire?
    lead_times = []
    for i in range(len(y_true_sequences)):
        true_seq = y_true_sequences[i]
        pred_seq = y_pred_sequences[i]

        # Find first true event
        true_event_idx = np.where(
            (true_seq < threshold_low) | (true_seq > threshold_high)
        )[0]
        pred_event_idx = np.where(
            (pred_seq < threshold_low) | (pred_seq > threshold_high)
        )[0]

        if len(true_event_idx) > 0 and len(pred_event_idx) > 0:
            lead = (true_event_idx[0] - pred_event_idx[0]) * sampling_interval_min
            if lead > 0:
                lead_times.append(lead)

    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "hypo_sensitivity": hypo_sens,
        "hypo_specificity": hypo_spec,
        "hyper_sensitivity": hyper_sens,
        "hyper_specificity": hyper_spec,
        "mean_lead_time_min": mean_lead,
        "n_samples": len(y_true_sequences),
    }


def full_evaluation(
    y_true: np.ndarray,
    y_pred_median: np.ndarray,
    y_true_sequences: Optional[np.ndarray] = None,
    y_pred_sequences: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
) -> EvaluationReport:
    """Run complete clinical evaluation suite.

    Args:
        y_true: Flattened true glucose values
        y_pred_median: Flattened median predictions
        y_true_sequences: [N, horizon] for event detection
        y_pred_sequences: [N, horizon] for event detection
        save_dir: Directory to save plots
    """
    mard = compute_mard(y_true, y_pred_median)
    rmse = compute_rmse(y_true, y_pred_median)
    mae = compute_mae(y_true, y_pred_median)
    clarke = clarke_error_grid_analysis(y_true, y_pred_median)

    # Event detection metrics
    event_metrics = {}
    if y_true_sequences is not None and y_pred_sequences is not None:
        event_metrics = compute_adverse_event_metrics(
            y_true_sequences, y_pred_sequences
        )

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plot_clarke_error_grid(
            y_true, y_pred_median,
            save_path=os.path.join(save_dir, "clarke_error_grid.png")
        )

    return EvaluationReport(
        mard=mard,
        rmse=rmse,
        mae=mae,
        clarke_zones=clarke,
        hypo_sensitivity=event_metrics.get("hypo_sensitivity", 0.0),
        hypo_specificity=event_metrics.get("hypo_specificity", 0.0),
        hyper_sensitivity=event_metrics.get("hyper_sensitivity", 0.0),
        hyper_specificity=event_metrics.get("hyper_specificity", 0.0),
        mean_lead_time_min=event_metrics.get("mean_lead_time_min", 0.0),
    )
