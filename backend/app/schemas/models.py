"""
DiaTwin API Schemas (Pydantic v2)

Request/response models for all API endpoints with strict validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# ---------------------------------------------------------------------------
# Patient Models
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """Static EHR patient profile."""
    patient_id: str = Field(..., description="Patient identifier (e.g., IND-0042)")
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    height_cm: float = Field(..., ge=100, le=220)
    weight_kg: float = Field(..., ge=30, le=200)
    bmi: float = Field(..., ge=15, le=60)
    waist_hip_ratio: float = Field(..., ge=0.5, le=1.5)
    hba1c: float = Field(..., ge=4.0, le=15.0, description="HbA1c (%)")
    fasting_glucose_mgdl: float = Field(..., ge=50, le=500)
    serum_creatinine_mgdl: float = Field(..., ge=0.3, le=10.0)
    egfr: float = Field(..., ge=5, le=150, description="eGFR (mL/min/1.73m²)")
    total_cholesterol: float = Field(..., ge=50, le=500)
    triglycerides: float = Field(..., ge=30, le=1000)
    hdl: float = Field(..., ge=10, le=120)
    ldl: float = Field(..., ge=20, le=400)
    disease_duration_years: float = Field(..., ge=0, le=50)
    has_hypertension: bool
    metformin: bool
    sglt2_inhibitor: bool
    sulfonylurea: bool
    insulin_basal: bool
    insulin_bolus: bool
    insulin_resistance_level: str = Field(..., pattern="^(low|moderate|high|severe)$")
    glycemic_variability: str = Field(..., pattern="^(low|moderate|high)$")


class PatientSummary(BaseModel):
    """Compact patient card for dashboard."""
    patient_id: str
    age: int
    sex: str
    bmi: float
    hba1c: float
    egfr: float
    insulin_resistance_level: str
    glycemic_variability: str
    medications: List[str]
    risk_level: str = Field(default="moderate", description="Overall risk assessment")


# ---------------------------------------------------------------------------
# Prediction Models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Request for glucose trajectory prediction."""
    patient_id: str
    lookback_readings: Optional[List[float]] = Field(
        None, description="Recent CGM readings (5-min intervals). If None, uses stored data."
    )
    horizon_steps: int = Field(
        default=24, ge=1, le=48, description="Number of future steps to predict"
    )


class QuantilePrediction(BaseModel):
    """Single time-step prediction with uncertainty."""
    time_offset_min: int = Field(..., description="Minutes from current time")
    q10: float = Field(..., description="10th percentile (lower bound)")
    q50: float = Field(..., description="50th percentile (median prediction)")
    q90: float = Field(..., description="90th percentile (upper bound)")


class PredictionResponse(BaseModel):
    """Trajectory prediction with confidence bands and alerts."""
    patient_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    current_glucose: float
    predictions: List[QuantilePrediction]
    alerts: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Clinical alerts (hypo/hyper warnings)"
    )
    time_in_range_pct: float = Field(
        default=0.0, description="Predicted time in 70-180 mg/dL range (%)"
    )


# ---------------------------------------------------------------------------
# Counterfactual Simulation Models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request for What-If counterfactual simulation."""
    patient_id: str
    current_glucose: float = Field(
        default=120.0, ge=30, le=500, description="Current glucose level (mg/dL)"
    )
    carb_grams: float = Field(
        default=0.0, ge=0, le=200, description="Carbohydrate intake (grams)"
    )
    exercise_duration_min: float = Field(
        default=0.0, ge=0, le=120, description="Exercise duration (minutes)"
    )
    exercise_intensity: float = Field(
        default=1.0, ge=0, le=2.0,
        description="Exercise intensity (0=light, 1=moderate, 2=vigorous)"
    )
    simulation_duration_min: float = Field(
        default=180.0, ge=30, le=360, description="Simulation window (minutes)"
    )


class SimulationResponse(BaseModel):
    """Counterfactual simulation results."""
    patient_id: str
    time_points: List[float] = Field(..., description="Time axis (minutes)")
    glucose_baseline: List[float] = Field(
        ..., description="Glucose trajectory without intervention"
    )
    glucose_intervention: List[float] = Field(
        ..., description="Glucose trajectory with intervention"
    )
    peak_reduction_mgdl: float = Field(
        ..., description="Maximum glucose reduction from intervention"
    )
    time_to_peak_min: float
    auc_reduction: float = Field(
        ..., description="Area-under-curve reduction (mg·min/dL)"
    )
    recommendation: str = Field(
        default="", description="Clinical recommendation text"
    )


# ---------------------------------------------------------------------------
# Alert Models
# ---------------------------------------------------------------------------

class AlertItem(BaseModel):
    """Clinical alert for the early warning feed."""
    patient_id: str
    alert_type: str = Field(..., description="hypo|hyper|spike|nocturnal")
    severity: str = Field(..., description="info|warning|critical")
    message: str
    predicted_value: float
    predicted_time_min: int = Field(
        ..., description="Minutes until predicted event"
    )
    current_glucose: float


class AlertFeed(BaseModel):
    """Prioritized patient alert queue."""
    alerts: List[AlertItem]
    total_patients: int
    patients_stable: int
    patients_at_risk: int


# ---------------------------------------------------------------------------
# Evaluation Models
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    """Model evaluation metrics."""
    mard: float = Field(..., description="Mean Absolute Relative Difference (%)")
    rmse: float = Field(..., description="Root Mean Squared Error (mg/dL)")
    mae: float = Field(..., description="Mean Absolute Error (mg/dL)")
    clarke_zones: Dict[str, float] = Field(
        ..., description="Clarke Error Grid zone percentages"
    )
    hypo_sensitivity: float
    hyper_sensitivity: float
    mean_lead_time_min: float
