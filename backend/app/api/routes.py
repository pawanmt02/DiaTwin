"""
DiaTwin API Routes

REST endpoints for:
  - Patient profiles and listings
  - Glucose trajectory prediction
  - Counterfactual intervention simulation
  - Alert feed generation
  - Model evaluation metrics
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from backend.app.schemas.models import (
    PatientProfile,
    PatientSummary,
    PredictionRequest,
    PredictionResponse,
    SimulationRequest,
    SimulationResponse,
    AlertItem,
    AlertFeed,
)
from backend.app.services.inference import inference_service

router = APIRouter()


# ---------------------------------------------------------------------------
# Patient Endpoints
# ---------------------------------------------------------------------------

@router.get("/patients")
async def list_patients(
    limit: int = Query(default=50, ge=1, le=500),
):
    """List all patients with summary cards."""
    summaries = inference_service.get_patient_list(limit=limit)
    # Transform to frontend-expected format
    result = []
    for s in summaries:
        result.append({
            "id": s["patient_id"],
            "name": s["patient_id"],  # Use ID as name for synthetic data
            "age": s["age"],
            "sex": "Male" if s["sex"] == "M" else "Female",
            "bmi": s["bmi"],
            "hba1c": s["hba1c"],
            "egfr": s["egfr"],
            "medications": s["medications"],
            "insulinResistanceLevel": s["insulin_resistance_level"].capitalize(),
            "riskLevel": s["risk_level"],
        })
    return result


@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get detailed patient profile."""
    profile = inference_service.get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    # Add computed fields
    meds = []
    if profile.get("metformin"): meds.append("Metformin 1000mg")
    if profile.get("sglt2_inhibitor"): meds.append("SGLT2 Inhibitor")
    if profile.get("sulfonylurea"): meds.append("Sulfonylurea")
    if profile.get("insulin_basal"): meds.append("Basal Insulin")
    if profile.get("insulin_bolus"): meds.append("Bolus Insulin")

    return {
        **profile,
        "medications_list": meds,
        "phenotype_summary": _compute_phenotype_summary(profile),
    }


@router.get("/patients/{patient_id}/cgm")
async def get_patient_cgm(
    patient_id: str,
    hours: int = Query(default=4, ge=1, le=168),
):
    """Get recent CGM data for a patient as frontend-compatible CGMData array."""
    cgm_df = inference_service.load_patient_cgm(patient_id)
    if cgm_df is None:
        raise HTTPException(
            status_code=404,
            detail=f"CGM data not found for patient {patient_id}"
        )

    # Return last N hours as historical data
    n_readings = hours * 12
    recent = cgm_df.tail(n_readings)

    result = []
    for _, row in recent.iterrows():
        result.append({
            "timestamp": str(row["timestamp"]),
            "value": round(float(row["glucose_mgdl"]), 1),
            "isPredicted": False,
        })

    # Append predictions from the model (last 2 hours forecast)
    try:
        pred = inference_service.predict_trajectory(
            patient_id=patient_id, horizon=24
        )
        from datetime import timedelta
        import pandas as pd
        last_ts = pd.to_datetime(str(recent.iloc[-1]["timestamp"]))
        for p in pred["predictions"]:
            future_ts = last_ts + timedelta(minutes=p["time_offset_min"])
            result.append({
                "timestamp": future_ts.isoformat(),
                "value": p["q50"],
                "isPredicted": True,
                "q10": p["q10"],
                "q50": p["q50"],
                "q90": p["q90"],
            })
    except Exception:
        pass  # Return historical only if prediction fails

    return result


# ---------------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict_trajectory(request: PredictionRequest):
    """Generate multi-horizon glucose trajectory forecast with uncertainty."""
    try:
        result = inference_service.predict_trajectory(
            patient_id=request.patient_id,
            recent_readings=request.lookback_readings,
            horizon=request.horizon_steps,
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/predict/{patient_id}", response_model=PredictionResponse)
async def predict_for_patient(
    patient_id: str,
    horizon: int = Query(default=24, ge=1, le=48),
):
    """Quick prediction using stored CGM data."""
    profile = inference_service.get_patient_profile(patient_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    try:
        result = inference_service.predict_trajectory(
            patient_id=patient_id,
            horizon=horizon,
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# Simulation Endpoints
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=SimulationResponse)
async def simulate_intervention(request: SimulationRequest):
    """Run counterfactual What-If simulation."""
    try:
        result = inference_service.simulate_intervention(
            patient_id=request.patient_id,
            current_glucose=request.current_glucose,
            carb_grams=request.carb_grams,
            exercise_duration_min=request.exercise_duration_min,
            exercise_intensity=request.exercise_intensity,
            simulation_duration_min=request.simulation_duration_min,
        )
        return SimulationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Alert Feed Endpoints
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=AlertFeed)
async def get_alert_feed(
    limit: int = Query(default=20, ge=1, le=100),
):
    """Get prioritized patient alert feed.

    Runs predictions for all patients and aggregates alerts.
    """
    patients = inference_service.get_patient_list(limit=limit)
    all_alerts = []
    stable_count = 0
    at_risk_count = 0

    for p in patients[:20]:  # Limit to 20 for performance
        try:
            result = inference_service.predict_trajectory(
                patient_id=p["patient_id"], horizon=12
            )
            if result["alerts"]:
                at_risk_count += 1
                for alert in result["alerts"][:2]:  # Max 2 alerts per patient
                    all_alerts.append(AlertItem(
                        patient_id=p["patient_id"],
                        alert_type=alert["type"],
                        severity=alert["severity"],
                        message=alert["message"],
                        predicted_value=result["predictions"][0]["q50"],
                        predicted_time_min=int(alert.get("time_offset_min", 0)),
                        current_glucose=result["current_glucose"],
                    ))
            else:
                stable_count += 1
        except Exception:
            pass

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    all_alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    return AlertFeed(
        alerts=all_alerts[:limit],
        total_patients=len(patients),
        patients_stable=stable_count,
        patients_at_risk=at_risk_count,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compute_phenotype_summary(profile: dict) -> str:
    """Generate clinical phenotype summary text."""
    parts = []

    bmi = float(profile.get("bmi", 0))
    if bmi > 30:
        parts.append("Obese (Class I)")
    elif bmi > 27:
        parts.append("Overweight")
    elif bmi > 23:
        parts.append("High Normal BMI (Asian criteria)")

    ir = profile.get("insulin_resistance_level", "")
    if ir == "severe":
        parts.append("Severe Insulin Resistance")
    elif ir == "high":
        parts.append("High Insulin Resistance")

    gv = profile.get("glycemic_variability", "")
    if gv == "high":
        parts.append("High Glycemic Variability")

    egfr = float(profile.get("egfr", 90))
    if egfr < 30:
        parts.append("Severe Renal Impairment")
    elif egfr < 60:
        parts.append("Moderate Renal Impairment")

    return " | ".join(parts) if parts else "Standard T2DM Profile"
