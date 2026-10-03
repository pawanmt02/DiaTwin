"""
DiaTwin Counterfactual Simulation Engine

Implements the Bergman Minimal Model for glucose-insulin dynamics,
enabling "What-If" simulations for:
  - Exercise interventions (walking, running)
  - Meal carbohydrate challenges
  - Insulin timing adjustments

The ODE system is solved numerically using scipy.integrate.solve_ivp.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from backend.app.core.config import config as cfg


@dataclass
class SimulationResult:
    """Container for counterfactual simulation output."""
    time_minutes: np.ndarray        # Time axis (minutes from t=0)
    glucose_baseline: np.ndarray    # Baseline glucose trajectory (no intervention)
    glucose_intervention: np.ndarray  # Modified glucose trajectory (with intervention)
    peak_reduction_mgdl: float      # Max glucose reduction from intervention
    time_to_peak_min: float         # Time of peak glucose reduction
    auc_reduction: float            # Area-under-curve reduction (mg/dL·min)


class MealModel:
    """Models carbohydrate absorption as a two-compartment gut model.

    Converts a meal of `carbs_grams` into a glucose appearance rate D(t)
    using a Rayleigh-like absorption curve calibrated for Indian dietary staples.
    """

    def __init__(
        self,
        carbs_grams: float = 60.0,
        absorption_rate: float = 0.03,
        peak_time_min: float = 45.0,
        bioavailability: float = 0.8,
    ):
        self.carbs_grams = carbs_grams
        self.absorption_rate = absorption_rate
        self.peak_time_min = peak_time_min
        self.bioavailability = bioavailability
        # Convert carbs to mg glucose (1g carb ≈ 1000mg glucose)
        self.total_glucose_mg = carbs_grams * 1000 * bioavailability

    def glucose_appearance_rate(self, t: float) -> float:
        """Glucose appearance rate D(t) in mg/dL/min.

        Uses a gamma-like function peaking at peak_time_min.
        Assumes distribution volume of ~140 dL for a 70kg adult.
        """
        if t < 0:
            return 0.0

        distribution_volume = 140.0  # dL (approximate for 70kg person)
        k = self.absorption_rate
        # Normalized gamma-like absorption
        rate = (
            self.total_glucose_mg / distribution_volume
            * k**2 * t * np.exp(-k * t)
        )
        return max(rate, 0.0)


class ExerciseModel:
    """Models exercise-induced glucose disposal enhancement.

    Exercise increases peripheral glucose uptake through:
    1. Insulin-independent GLUT4 translocation
    2. Enhanced insulin sensitivity (post-exercise)
    """

    def __init__(
        self,
        start_min: float = 0.0,
        duration_min: float = 20.0,
        intensity: float = 1.0,  # 0-2 scale (0=light, 1=moderate, 2=vigorous)
    ):
        self.start_min = start_min
        self.duration_min = duration_min
        self.intensity = intensity

    def activity_signal(self, t: float) -> float:
        """Activity intensity signal A(t).

        Returns a smooth bell-shaped exercise profile with lingering
        post-exercise effects (enhanced insulin sensitivity).
        """
        if t < self.start_min:
            return 0.0

        t_exercise = t - self.start_min

        if t_exercise <= self.duration_min:
            # Active exercise phase (ramps up then plateaus)
            ramp = min(t_exercise / 5.0, 1.0)  # 5-min ramp-up
            return self.intensity * ramp
        else:
            # Post-exercise decay (effects linger for ~30-60 min)
            t_post = t_exercise - self.duration_min
            decay = np.exp(-t_post / 30.0)  # 30-min half-life
            return self.intensity * 0.5 * decay


class BergmanMinimalModel:
    """Bergman Minimal Model for glucose-insulin dynamics.

    System of ODEs:
        dG/dt = -(p1 + X) * G(t) + p1 * Gb + D(t)
        dX/dt = -p2 * X(t) + p3 * (I(t) - Ib) - γ_exercise * A(t)

    Where:
        G(t)  = plasma glucose concentration (mg/dL)
        X(t)  = insulin action in remote compartment (1/min)
        I(t)  = plasma insulin concentration (mU/L)
        D(t)  = meal glucose appearance rate (mg/dL/min)
        A(t)  = exercise activity signal
        Gb    = basal glucose
        Ib    = basal insulin
        p1    = glucose effectiveness (1/min)
        p2    = insulin action decay rate (1/min)
        p3    = insulin sensitivity (1/min per mU/L)
        γ     = exercise sensitivity coefficient
    """

    def __init__(
        self,
        p1: float = None,
        p2: float = None,
        p3: float = None,
        gb: float = None,
        ib: float = None,
        exercise_sensitivity: float = None,
        insulin_resistance_factor: float = 1.0,
    ):
        """
        Args:
            insulin_resistance_factor: Multiplier for insulin sensitivity.
                >1 means more resistant (slower insulin action).
                Derived from patient's static EHR insulin resistance level.
        """
        self.p1 = p1 or cfg.clinical.bergman_p1
        self.p2 = p2 or cfg.clinical.bergman_p2
        self.p3 = (p3 or cfg.clinical.bergman_p3) / insulin_resistance_factor
        self.gb = gb or cfg.clinical.bergman_gb
        self.ib = ib or cfg.clinical.bergman_ib
        self.gamma = exercise_sensitivity or cfg.clinical.exercise_sensitivity
        self.insulin_resistance_factor = insulin_resistance_factor

    def _ode_system(
        self,
        t: float,
        state: np.ndarray,
        meal: Optional[MealModel],
        exercise: Optional[ExerciseModel],
        insulin_profile: callable,
    ) -> np.ndarray:
        """ODE right-hand side for the Bergman model."""
        G, X = state

        # Glucose appearance from meal
        D = meal.glucose_appearance_rate(t) if meal else 0.0

        # Activity signal
        A = exercise.activity_signal(t) if exercise else 0.0

        # Insulin profile (simplified: basal + meal-triggered bolus)
        I = insulin_profile(t) if insulin_profile else self.ib

        # ODEs
        dG_dt = -(self.p1 + X) * G + self.p1 * self.gb + D
        dX_dt = -self.p2 * X + self.p3 * (I - self.ib) + self.gamma * A

        return np.array([dG_dt, dX_dt])

    def simulate(
        self,
        initial_glucose: float = 110.0,
        duration_min: float = 180.0,
        meal: Optional[MealModel] = None,
        exercise: Optional[ExerciseModel] = None,
        insulin_profile: Optional[callable] = None,
        dt: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate glucose trajectory.

        Args:
            initial_glucose: Starting glucose (mg/dL)
            duration_min: Simulation duration (minutes)
            meal: MealModel for carbohydrate challenge
            exercise: ExerciseModel for physical activity
            insulin_profile: Callable(t) → insulin concentration
            dt: Output time resolution (minutes)

        Returns:
            (time_array, glucose_array) both in minutes and mg/dL
        """
        t_span = (0, duration_min)
        t_eval = np.arange(0, duration_min + dt, dt)
        y0 = [initial_glucose, 0.0]  # [G(0), X(0)]

        # Default insulin profile: constant basal
        if insulin_profile is None:
            insulin_profile = lambda t: self.ib

        sol = solve_ivp(
            fun=lambda t, y: self._ode_system(t, y, meal, exercise, insulin_profile),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            max_step=1.0,
        )

        # Clamp glucose to physiological range
        glucose = np.clip(sol.y[0], 30, 500)

        return sol.t, glucose


class CounterfactualEngine:
    """High-level counterfactual simulation interface.

    Runs paired simulations (baseline vs. intervention) and computes
    the differential impact for the clinician dashboard.
    """

    def __init__(self, insulin_resistance_factor: float = 1.0):
        self.model = BergmanMinimalModel(
            insulin_resistance_factor=insulin_resistance_factor
        )

    @staticmethod
    def ir_factor_from_level(level: Optional[str]) -> float:
        """Map insulin resistance level to numerical factor."""
        mapping = {
            "low": 0.8,
            "moderate": 1.2,
            "high": 1.8,
            "severe": 2.5,
        }
        if level is None:
            return 1.0
        normalized = str(level).strip().lower()
        return mapping.get(normalized, 1.0)

    def simulate_meal_with_exercise(
        self,
        initial_glucose: float = 110.0,
        carb_grams: float = 60.0,
        exercise_duration_min: float = 0.0,
        exercise_start_min: float = 30.0,
        exercise_intensity: float = 1.0,
        duration_min: float = 180.0,
    ) -> SimulationResult:
        """Simulate meal scenario with optional exercise intervention.

        Returns baseline (meal only) and intervention (meal + exercise) trajectories.
        """
        meal = MealModel(carbs_grams=carb_grams)

        # Baseline: meal without exercise
        t_base, g_base = self.model.simulate(
            initial_glucose=initial_glucose,
            duration_min=duration_min,
            meal=meal,
            exercise=None,
        )

        # Intervention: meal + exercise
        if exercise_duration_min > 0:
            exercise = ExerciseModel(
                start_min=exercise_start_min,
                duration_min=exercise_duration_min,
                intensity=exercise_intensity,
            )
            t_int, g_int = self.model.simulate(
                initial_glucose=initial_glucose,
                duration_min=duration_min,
                meal=meal,
                exercise=exercise,
            )
        else:
            t_int, g_int = t_base, g_base

        # Compute impact metrics
        diff = g_base - g_int
        peak_reduction = float(np.max(diff))
        peak_time = float(t_base[np.argmax(diff)])
        auc_reduction = float(np.trapz(diff, t_base))

        return SimulationResult(
            time_minutes=t_base,
            glucose_baseline=g_base,
            glucose_intervention=g_int,
            peak_reduction_mgdl=peak_reduction,
            time_to_peak_min=peak_time,
            auc_reduction=auc_reduction,
        )

    def simulate_exercise_only(
        self,
        initial_glucose: float = 150.0,
        exercise_duration_min: float = 20.0,
        exercise_intensity: float = 1.0,
        duration_min: float = 120.0,
    ) -> SimulationResult:
        """Simulate exercise intervention on current glucose level."""
        # Baseline: no exercise
        t_base, g_base = self.model.simulate(
            initial_glucose=initial_glucose,
            duration_min=duration_min,
        )

        # Intervention: exercise
        exercise = ExerciseModel(
            start_min=0.0,
            duration_min=exercise_duration_min,
            intensity=exercise_intensity,
        )
        t_int, g_int = self.model.simulate(
            initial_glucose=initial_glucose,
            duration_min=duration_min,
            exercise=exercise,
        )

        diff = g_base - g_int
        peak_reduction = float(np.max(diff))
        peak_time = float(t_base[np.argmax(diff)])

        return SimulationResult(
            time_minutes=t_base,
            glucose_baseline=g_base,
            glucose_intervention=g_int,
            peak_reduction_mgdl=peak_reduction,
            time_to_peak_min=peak_time,
            auc_reduction=float(np.trapz(diff, t_base)),
        )
