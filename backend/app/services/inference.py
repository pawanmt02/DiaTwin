"""
DiaTwin Inference Service

Provides high-level prediction and simulation interfaces used by API endpoints.
Handles model loading, data preparation, and result formatting.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

from backend.app.core.config import config as cfg
from backend.app.core.preprocessing import (
    StaticEHRPreprocessor,
    DynamicCGMPreprocessor,
)
from backend.app.models.forecaster import DiaTwinForecaster, build_model
from backend.app.services.counterfactual import CounterfactualEngine


class InferenceService:
    """Singleton service managing model inference and simulation.

    Loads the trained model and preprocessors, caches patient data,
    and provides prediction/simulation methods for the API layer.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[DiaTwinForecaster] = None
        self.ehr_preprocessor = StaticEHRPreprocessor()
        self.cgm_preprocessor = DynamicCGMPreprocessor()
        self.patient_profiles: Dict[str, dict] = {}
        self.patient_cgm_cache: Dict[str, pd.DataFrame] = {}
        self._loaded = False

    def load(self, model_path: Optional[str] = None):
        """Load model, preprocessors, and patient data."""
        # Load patient profiles
        ehr_path = cfg.data.ehr_profiles_path
        if os.path.exists(ehr_path):
            df = pd.read_csv(ehr_path)
            for _, row in df.iterrows():
                self.patient_profiles[row["patient_id"]] = row.to_dict()

            # Fit preprocessor on available data
            self.ehr_preprocessor.fit(df)

        # Load model if checkpoint exists
        model_path = model_path or cfg.model.best_model_path
        if os.path.exists(model_path):
            # Determine static dim from preprocessor
            static_dim = len(cfg.data.static_numerical_features) + len(
                cfg.data.static_categorical_features
            )
            self.model = build_model(
                static_dim=static_dim,
                dynamic_dim=len(cfg.data.dynamic_features),
                device=self.device,
            )
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            # Build untrained model for demo/simulation mode
            static_dim = len(cfg.data.static_numerical_features) + len(
                cfg.data.static_categorical_features
            )
            self.model = build_model(
                static_dim=static_dim,
                dynamic_dim=len(cfg.data.dynamic_features),
                device=self.device,
            )
            self.model.eval()
            print("No trained model found — running in simulation/demo mode")

        self._loaded = True

    def get_patient_profile(self, patient_id: str) -> Optional[dict]:
        """Retrieve patient profile by ID."""
        return self.patient_profiles.get(patient_id)

    def get_patient_list(self, limit: int = 50) -> List[dict]:
        """Return list of patient summaries."""
        profiles = list(self.patient_profiles.values())[:limit]
        summaries = []
        for p in profiles:
            meds = []
            if p.get("metformin"): meds.append("Metformin")
            if p.get("sglt2_inhibitor"): meds.append("SGLT2i")
            if p.get("sulfonylurea"): meds.append("Sulfonylurea")
            if p.get("insulin_basal"): meds.append("Basal Insulin")
            if p.get("insulin_bolus"): meds.append("Bolus Insulin")

            # Risk level from HbA1c
            hba1c = p.get("hba1c", 7.0)
            if hba1c > 10:
                risk = "critical"
            elif hba1c > 8.5:
                risk = "high"
            elif hba1c > 7.0:
                risk = "moderate"
            else:
                risk = "low"

            summaries.append({
                "patient_id": p.get("patient_id", ""),
                "age": int(p.get("age", 0)),
                "sex": p.get("sex", ""),
                "bmi": round(float(p.get("bmi", 0)), 1),
                "hba1c": round(float(p.get("hba1c", 0)), 1),
                "egfr": round(float(p.get("egfr", 0)), 1),
                "insulin_resistance_level": p.get("insulin_resistance_level", ""),
                "glycemic_variability": p.get("glycemic_variability", ""),
                "medications": meds,
                "risk_level": risk,
            })
        return summaries

    def load_patient_cgm(self, patient_id: str) -> Optional[pd.DataFrame]:
        """Load CGM data for a patient (with caching)."""
        if patient_id in self.patient_cgm_cache:
            return self.patient_cgm_cache[patient_id]

        cgm_path = os.path.join(cfg.data.cgm_data_dir, f"{patient_id}.csv")
        if os.path.exists(cgm_path):
            df = pd.read_csv(cgm_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.patient_cgm_cache[patient_id] = df
            return df
        return None

    @torch.no_grad()
    def predict_trajectory(
        self,
        patient_id: str,
        recent_readings: Optional[List[float]] = None,
        horizon: int = 24,
    ) -> Dict:
        """Generate multi-quantile glucose trajectory forecast.

        If recent_readings is provided, uses those directly.
        Otherwise, loads the last `lookback` readings from stored CGM data.
        """
        if not self._loaded:
            self.load()

        profile = self.get_patient_profile(patient_id)
        lookback = cfg.model.lookback_window

        # Prepare dynamic input
        if recent_readings and len(recent_readings) >= lookback:
            glucose_vals = np.array(recent_readings[-lookback:])
        else:
            # Load from stored CGM
            cgm_df = self.load_patient_cgm(patient_id)
            if cgm_df is not None and len(cgm_df) >= lookback:
                # Take the last day's data, pick a random window
                glucose_vals = cgm_df["glucose_mgdl"].values[-lookback:]
            else:
                # Generate synthetic demo data
                np.random.seed(hash(patient_id) % 2**31)
                base = float(profile["fasting_glucose_mgdl"]) if profile else 120
                glucose_vals = base + np.cumsum(np.random.randn(lookback) * 3)
                glucose_vals = np.clip(glucose_vals, 50, 350)

        # Build feature matrix
        current_glucose = float(glucose_vals[-1])
        delta = np.diff(glucose_vals, prepend=glucose_vals[0])
        delta2 = np.diff(delta, prepend=delta[0])

        # Synthetic step/HR data if not available
        steps_15 = np.random.poisson(50, lookback).astype(float)
        steps_30 = np.convolve(steps_15, np.ones(2), mode="same")
        steps_60 = np.convolve(steps_15, np.ones(4), mode="same")
        hr = np.random.normal(75, 5, lookback)

        # Time encoding (assume evenly spaced)
        t_idx = np.arange(lookback) * 5  # minutes
        t_sin = np.sin(2 * np.pi * t_idx / 1440)
        t_cos = np.cos(2 * np.pi * t_idx / 1440)

        dynamic_features = np.column_stack([
            glucose_vals, delta, delta2,
            steps_15, steps_30, steps_60,
            hr, t_sin, t_cos
        ]).astype(np.float32)

        # Static features
        if profile:
            df_profile = pd.DataFrame([profile])
            static_vec = self.ehr_preprocessor.transform(df_profile)[0]
        else:
            static_vec = np.zeros(
                len(cfg.data.static_numerical_features) +
                len(cfg.data.static_categorical_features),
                dtype=np.float32
            )

        # Tensors
        x_static = torch.tensor(static_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_dynamic = torch.tensor(dynamic_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        predictions = self.model(x_static, x_dynamic)  # [1, H, 3]
        preds = predictions[0].cpu().numpy()  # [H, 3]

        # Format results
        pred_list = []
        alerts = []

        for k in range(min(horizon, preds.shape[0])):
            t_offset = (k + 1) * cfg.data.sampling_interval_min
            q10, q50, q90 = float(preds[k, 0]), float(preds[k, 1]), float(preds[k, 2])

            pred_list.append({
                "time_offset_min": t_offset,
                "q10": round(q10, 1),
                "q50": round(q50, 1),
                "q90": round(q90, 1),
            })

            # Alert generation
            if q50 > cfg.clinical.hyperglycemia_threshold:
                alerts.append({
                    "type": "hyperglycemia",
                    "severity": "critical" if q50 > cfg.clinical.severe_hyperglycemia_threshold else "warning",
                    "message": f"Predicted spike to {q50:.0f} mg/dL at t+{t_offset}min",
                    "time_offset_min": str(t_offset),
                })
            elif q50 < cfg.clinical.hypoglycemia_threshold:
                alerts.append({
                    "type": "hypoglycemia",
                    "severity": "critical" if q50 < cfg.clinical.severe_hypoglycemia_threshold else "warning",
                    "message": f"Predicted drop to {q50:.0f} mg/dL at t+{t_offset}min",
                    "time_offset_min": str(t_offset),
                })

        # Time in range
        q50_values = [p["q50"] for p in pred_list]
        in_range = sum(1 for v in q50_values
                       if cfg.clinical.target_range_low <= v <= cfg.clinical.target_range_high)
        tir = round(in_range / max(len(q50_values), 1) * 100, 1)

        return {
            "patient_id": patient_id,
            "current_glucose": round(current_glucose, 1),
            "predictions": pred_list,
            "alerts": alerts,
            "time_in_range_pct": tir,
        }

    def simulate_intervention(
        self,
        patient_id: str,
        current_glucose: float = 120.0,
        carb_grams: float = 0.0,
        exercise_duration_min: float = 0.0,
        exercise_intensity: float = 1.0,
        simulation_duration_min: float = 180.0,
    ) -> Dict:
        """Run counterfactual What-If simulation."""
        profile = self.get_patient_profile(patient_id)

        # Patient-specific insulin resistance
        ir_level = profile.get("insulin_resistance_level", "moderate") if profile else "moderate"
        ir_factor = CounterfactualEngine.ir_factor_from_level(ir_level)

        engine = CounterfactualEngine(insulin_resistance_factor=ir_factor)

        result = engine.simulate_meal_with_exercise(
            initial_glucose=current_glucose,
            carb_grams=carb_grams,
            exercise_duration_min=exercise_duration_min,
            exercise_start_min=30.0,  # Start exercise 30 min after meal
            exercise_intensity=exercise_intensity,
            duration_min=simulation_duration_min,
        )

        # Sample every 5 minutes for efficiency
        step = 5
        indices = np.arange(0, len(result.time_minutes), step)
        if indices[-1] != len(result.time_minutes) - 1:
            indices = np.append(indices, len(result.time_minutes) - 1)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            result.peak_reduction_mgdl,
            exercise_duration_min,
            carb_grams,
            current_glucose,
        )

        return {
            "patient_id": patient_id,
            "time_points": result.time_minutes[indices].tolist(),
            "glucose_baseline": np.round(result.glucose_baseline[indices], 1).tolist(),
            "glucose_intervention": np.round(result.glucose_intervention[indices], 1).tolist(),
            "peak_reduction_mgdl": round(result.peak_reduction_mgdl, 1),
            "time_to_peak_min": round(result.time_to_peak_min, 0),
            "auc_reduction": round(result.auc_reduction, 0),
            "recommendation": recommendation,
        }

    @staticmethod
    def _generate_recommendation(
        peak_reduction: float,
        exercise_min: float,
        carb_grams: float,
        current_glucose: float,
    ) -> str:
        """Generate clinical recommendation text based on simulation."""
        parts = []

        if exercise_min > 0 and peak_reduction > 10:
            parts.append(
                f"A {exercise_min:.0f}-minute walk is projected to reduce peak glucose "
                f"by approximately {peak_reduction:.0f} mg/dL."
            )
            if peak_reduction > 30:
                parts.append("This is a significant attenuation of the postprandial spike.")

        if carb_grams > 80:
            parts.append(
                "Consider reducing carbohydrate portion size to limit "
                "postprandial excursion amplitude."
            )

        if current_glucose > 200:
            parts.append(
                "Current glucose is elevated. Monitor closely and consider "
                "pharmacological intervention if sustained."
            )

        if not parts:
            parts.append("Glucose trajectory appears stable within acceptable parameters.")

        parts.append(
            "⚠️ DiaTwin is an investigational decision-support prototype. "
            "Treatment adjustments must be verified with licensed medical judgment."
        )

        return " ".join(parts)


# Global service instance
inference_service = InferenceService()
