import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta

def simulate_meal_response(t, peak_time, duration, carb_load, ir_factor):
    """ Simulate glucose rise after a meal using a skewed gamma-like curve """
    if t < 0 or t > duration:
        return 0.0
    
    t_norm = t / peak_time
    
    if t <= peak_time:
        rise = (carb_load * ir_factor * 0.8) * np.sin((np.pi / 2) * t_norm)
        return rise
    else:
        decay_time = duration - peak_time
        t_decay = (t - peak_time) / decay_time
        rise = (carb_load * ir_factor * 0.8) * np.exp(-3 * t_decay)
        return rise

def generate_patient_cgm(patient, start_date):
    n_steps = 2016
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_steps)]
    
    pid = patient['patient_id']
    base_gluc = patient['fasting_glucose_mgdl']
    ir_level = patient['insulin_resistance_level']
    
    ir_factor_map = {'low': 1.0, 'moderate': 1.3, 'high': 1.6, 'severe': 2.0}
    ir_factor = ir_factor_map.get(ir_level, 1.3)
    
    on_insulin = patient['insulin_basal'] or patient['insulin_bolus']
    on_su = patient['sulfonylurea']
    
    glucose = np.full(n_steps, base_gluc, dtype=float)
    steps_5min = np.zeros(n_steps, dtype=int)
    heart_rate = np.full(n_steps, 75, dtype=int)
    
    for day in range(7):
        day_offset = day * 288
        
        # Dawn phenomenon
        dawn_start = day_offset + (4 * 12)
        dawn_end = day_offset + (6 * 12)
        dawn_rise = np.random.uniform(5, 15) * ir_factor
        for i in range(dawn_start, dawn_end):
            if i < n_steps:
                glucose[i:day_offset+288] += dawn_rise / (dawn_end - dawn_start)
            
        # Meals
        meals = [
            (int(np.random.normal(8*12, 6)), np.random.uniform(40, 70)),
            (int(np.random.normal(13*12, 6)), np.random.uniform(60, 100)),
            (int(np.random.normal(20*12, 6)), np.random.uniform(50, 80))
        ]
        
        for meal_idx, (meal_time, carb_load) in enumerate(meals):
            if np.random.rand() < 0.05:
                continue
                
            peak_time = int(np.random.uniform(45, 90) / 5)
            duration = int(np.random.uniform(120, 240) / 5)
            
            for t_meal in range(duration):
                idx = day_offset + meal_time + t_meal
                if 0 <= idx < n_steps:
                    resp = simulate_meal_response(t_meal, peak_time, duration, carb_load, ir_factor)
                    glucose[idx] += resp
                    
        # Exercise
        n_exercise = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        for _ in range(n_exercise):
            ex_start = day_offset + int(np.random.uniform(7*12, 19*12))
            ex_dur = int(np.random.uniform(20, 60) / 5)
            ex_drop = np.random.uniform(15, 40)
            
            for t_ex in range(ex_dur):
                idx = ex_start + t_ex
                if 0 <= idx < n_steps:
                    glucose[idx] -= ex_drop * (t_ex / ex_dur)
                    steps_5min[idx] += int(np.random.uniform(50, 200))
                    heart_rate[idx] = int(np.random.uniform(100, 150))
                    
        # Hypoglycemia
        if (on_insulin or on_su) and np.random.rand() < 0.1:
            hypo_start = day_offset + int(np.random.uniform(0, 288))
            hypo_dur = int(np.random.uniform(30, 60) / 5)
            for t_h in range(hypo_dur):
                idx = hypo_start + t_h
                if 0 <= idx < n_steps:
                    glucose[idx] = max(40, glucose[idx] - 30)
                    
        # Steps and HR
        for i in range(288):
            idx = day_offset + i
            if idx >= n_steps:
                break
            
            hr = i / 12
            if hr < 6 or hr > 22:
                steps_5min[idx] = 0
                heart_rate[idx] = max(50, min(65, heart_rate[idx] if heart_rate[idx] > 75 else int(np.random.uniform(50, 65))))
            else:
                if steps_5min[idx] == 0:
                    steps_5min[idx] = int(np.random.uniform(0, 50))
                if heart_rate[idx] == 75:
                    heart_rate[idx] = int(np.random.uniform(60, 90))
                    
    glucose += np.random.normal(0, 3, n_steps)
    glucose = np.clip(np.round(glucose), 40, 400)
    
    df = pd.DataFrame({
        'patient_id': pid,
        'timestamp': timestamps,
        'glucose_mgdl': glucose,
        'steps_5min': steps_5min,
        'heart_rate_bpm': heart_rate
    })
    
    return df

def generate_all_cgm(ehr_file="data/synthetic/ehr_profiles.csv", output_dir="data/synthetic/cgm/"):
    np.random.seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ehr_path = os.path.join(project_root, ehr_file)
    output_path = os.path.join(project_root, output_dir)
    
    os.makedirs(output_path, exist_ok=True)
    
    if not os.path.exists(ehr_path):
        print(f"Error: {ehr_path} not found. Run generate_synthetic_ehr.py first.")
        return
        
    print(f"Loading EHR profiles from {ehr_path}...")
    ehr_df = pd.read_csv(ehr_path)
    
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    
    print(f"Generating 7 days of 5-min CGM data for {len(ehr_df)} patients...")
    for _, patient in tqdm(ehr_df.iterrows(), total=len(ehr_df)):
        patient_df = generate_patient_cgm(patient, start_date)
        patient_df.to_csv(os.path.join(output_path, f"{patient['patient_id']}.csv"), index=False)
        
    print(f"\nSaved CGM streams to {output_path}")
    print("\nSample Data Summary (first patient):")
    sample_df = pd.read_csv(os.path.join(output_path, f"{ehr_df.iloc[0]['patient_id']}.csv"))
    print(sample_df.describe())

if __name__ == "__main__":
    generate_all_cgm()
