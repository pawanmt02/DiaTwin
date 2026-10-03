"""
DiaTwin Configuration Module

Central configuration for all system parameters including model hyperparameters,
data pipeline settings, clinical thresholds, and API configuration.
"""

from dataclasses import dataclass, field
from typing import List
import os


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def resolve_project_path(path: str) -> str:
    """Convert repo-relative paths to absolute paths anchored at the project root."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    # Paths
    raw_data_dir: str = field(default_factory=lambda: resolve_project_path("data/raw"))
    processed_data_dir: str = field(default_factory=lambda: resolve_project_path("data/processed"))
    synthetic_data_dir: str = field(default_factory=lambda: resolve_project_path("data/synthetic"))
    ehr_profiles_path: str = field(default_factory=lambda: resolve_project_path("data/synthetic/ehr_profiles.csv"))
    cgm_data_dir: str = field(default_factory=lambda: resolve_project_path("data/synthetic/cgm"))

    # CGM parameters
    sampling_interval_min: int = 5
    readings_per_hour: int = 12
    readings_per_day: int = 288

    # Static EHR features
    static_numerical_features: List[str] = field(default_factory=lambda: [
        "age", "height_cm", "weight_kg", "bmi", "waist_hip_ratio",
        "hba1c", "fasting_glucose_mgdl", "serum_creatinine_mgdl",
        "egfr", "total_cholesterol", "triglycerides", "hdl", "ldl",
        "disease_duration_years"
    ])
    static_categorical_features: List[str] = field(default_factory=lambda: [
        "sex", "has_hypertension", "metformin", "sglt2_inhibitor",
        "sulfonylurea", "insulin_basal", "insulin_bolus",
        "insulin_resistance_level", "glycemic_variability"
    ])

    # Dynamic features
    dynamic_features: List[str] = field(default_factory=lambda: [
        "glucose_mgdl", "delta_glucose", "delta2_glucose",
        "steps_15min", "steps_30min", "steps_60min",
        "heart_rate_bpm", "time_sin", "time_cos"
    ])


@dataclass
class ModelConfig:
    """Configuration for the predictive model."""
    # Architecture
    static_input_dim: int = 14      # Number of numerical static features
    static_embedding_dim: int = 32  # Dense projection of static features
    categorical_embedding_dim: int = 4  # Per-category embedding dim
    dynamic_input_dim: int = 9      # Number of dynamic input features
    hidden_dim: int = 128           # GRU hidden state dimension
    num_gru_layers: int = 2         # Stacked GRU layers
    num_attention_heads: int = 4    # Multi-head attention
    dropout: float = 0.2

    # Sequence parameters
    lookback_window: int = 24       # Past 2 hours (24 × 5 min)
    forecast_horizon: int = 24      # Future 2 hours (24 × 5 min)

    # Quantiles for uncertainty estimation
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 20
    early_stopping_patience: int = 5
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

    # Checkpoint
    checkpoint_dir: str = field(default_factory=lambda: resolve_project_path("models/checkpoints"))
    best_model_path: str = field(default_factory=lambda: resolve_project_path("models/checkpoints/best_model.pt"))


@dataclass
class ClinicalConfig:
    """Clinical thresholds and parameters."""
    # Glucose thresholds (mg/dL)
    hypoglycemia_threshold: float = 70.0
    hyperglycemia_threshold: float = 180.0
    severe_hypoglycemia_threshold: float = 54.0
    severe_hyperglycemia_threshold: float = 250.0
    target_range_low: float = 70.0
    target_range_high: float = 180.0

    # Alert lead time (minutes)
    minimum_alert_lead_time: int = 30
    target_alert_lead_time: int = 45

    # Bergman Minimal Model defaults
    bergman_p1: float = 0.028         # Glucose effectiveness (1/min)
    bergman_p2: float = 0.025         # Rate of insulin action decline (1/min)
    bergman_p3: float = 5.0e-6        # Insulin sensitivity (1/min per mU/L)
    bergman_gb: float = 100.0         # Basal glucose (mg/dL)
    bergman_ib: float = 10.0          # Basal insulin (mU/L)
    exercise_sensitivity: float = 0.015  # Exercise glucose reduction coefficient


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ])
    api_prefix: str = "/api/v1"


@dataclass
class DiaTwinConfig:
    """Master configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Global singleton
config = DiaTwinConfig()
