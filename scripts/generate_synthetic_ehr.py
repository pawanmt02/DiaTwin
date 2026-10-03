import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import scipy.stats as stats

def generate_ehr_data(num_patients=1000, output_file="data/synthetic/ehr_profiles.csv"):
    np.random.seed(42)
    
    # Determine the project root assuming the script is in DiaTwin/scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_path = os.path.join(project_root, output_file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating {num_patients} Indian-calibrated T2DM patient profiles...")
    
    # Base demographics
    patient_id = [f"IND-{str(i).zfill(4)}" for i in range(1, num_patients + 1)]
    
    # Age: 40-75, weighted toward 50-65
    age = np.clip(np.round(stats.beta.rvs(a=2, b=2, loc=35, scale=45, size=num_patients)), 40, 75)
    
    # Sex: M/F, ~55% M
    is_male = np.random.rand(num_patients) < 0.55
    sex = np.where(is_male, 'M', 'F')
    
    # Height: M ~167cm, F ~155cm, SD ~7
    height_cm = np.where(is_male, 
                         np.random.normal(167, 7, num_patients), 
                         np.random.normal(155, 7, num_patients))
    height_cm = np.round(height_cm, 1)
    
    # BMI: Asian Indian Phenotype: 22-35, mean ~27, skewed higher
    bmi = np.clip(stats.gamma.rvs(a=5, scale=1.2, loc=20, size=num_patients), 22, 45)
    bmi = np.round(bmi, 1)
    
    # Weight derived from BMI
    weight_kg = np.round(bmi * ((height_cm / 100) ** 2), 1)
    
    # Waist-hip ratio: M: 0.85-1.05, F: 0.80-0.95
    bmi_norm = (bmi - bmi.min()) / (bmi.max() - bmi.min())
    whr_m = np.random.uniform(0.85, 0.95, num_patients) + (bmi_norm * 0.1)
    whr_f = np.random.uniform(0.80, 0.88, num_patients) + (bmi_norm * 0.07)
    waist_hip_ratio = np.where(is_male, whr_m, whr_f)
    waist_hip_ratio = np.round(np.clip(waist_hip_ratio, 0.8, 1.1), 2)
    
    # HbA1c: 6.5-12.0%, mean ~8.2
    hba1c_base = np.random.normal(7.5, 1.0, num_patients)
    hba1c = hba1c_base + (bmi_norm * 1.5) + ((age - 50)/50)
    hba1c = np.round(np.clip(hba1c, 6.5, 13.0), 1)
    
    # Fasting Glucose: 100-300
    fasting_glucose_mgdl = np.round((28.7 * hba1c - 46.7) * np.random.uniform(0.8, 1.0, num_patients))
    fasting_glucose_mgdl = np.clip(fasting_glucose_mgdl, 100, 350)
    
    # Disease duration: 1-25
    disease_duration_years = np.round(np.random.gamma(2, 4, num_patients))
    disease_duration_years = np.clip(disease_duration_years, 1, 30)
    
    # eGFR & Creatinine
    egfr_base = 120 - (age - 40) * 0.8 - (disease_duration_years * 1.2) - ((hba1c - 6.5) * 2) + np.random.normal(0, 10, num_patients)
    egfr = np.round(np.clip(egfr_base, 15, 130), 1)
    
    serum_creatinine_mgdl = np.round(175 / (egfr * np.where(is_male, 1.0, 0.742)) ** (1/1.154), 2)
    serum_creatinine_mgdl = np.clip(serum_creatinine_mgdl, 0.5, 5.0)
    
    # Lipids
    total_cholesterol = np.round(np.random.normal(200, 40, num_patients) + (bmi_norm * 30))
    total_cholesterol = np.clip(total_cholesterol, 130, 350)
    
    triglycerides = np.round(np.random.gamma(3, 50, num_patients) + (hba1c * 10))
    triglycerides = np.clip(triglycerides, 80, 600)
    
    hdl_base = np.where(is_male, np.random.normal(45, 10, num_patients), np.random.normal(50, 10, num_patients))
    hdl = np.round(hdl_base - (bmi_norm * 10))
    hdl = np.clip(hdl, 25, 80)
    
    ldl = np.round(total_cholesterol - hdl - (triglycerides / 5))
    ldl = np.clip(ldl, 50, 250)
    
    # Comorbidities & Meds
    has_hypertension = np.random.rand(num_patients) < (0.3 + (age/200) + (bmi_norm * 0.2))
    metformin = np.random.rand(num_patients) < 0.80
    
    sglt2_prob = np.where(egfr > 45, 0.3, 0.05)
    sglt2_inhibitor = np.random.rand(num_patients) < sglt2_prob
    
    su_prob = np.where(hba1c > 7.5, 0.45, 0.2)
    sulfonylurea = np.random.rand(num_patients) < su_prob
    
    insulin_basal_prob = np.where((hba1c > 8.5) & (disease_duration_years > 5), 0.5, 0.1)
    insulin_basal = np.random.rand(num_patients) < insulin_basal_prob
    
    insulin_bolus_prob = np.where(insulin_basal & (hba1c > 9.0), 0.4, 0.02)
    insulin_bolus = np.random.rand(num_patients) < insulin_bolus_prob
    
    # Derived categorical
    def calc_ir_level(b, h, w):
        score = (b/25) + (h/7) + (w/0.9)
        if score < 3.0: return 'low'
        elif score < 3.5: return 'moderate'
        elif score < 4.0: return 'high'
        else: return 'severe'
    
    insulin_resistance_level = [calc_ir_level(b, h, w) for b, h, w in zip(bmi, hba1c, waist_hip_ratio)]
    
    def calc_gv_level(h, d):
        score = (h/7) + (d/10)
        if score < 2.0: return 'low'
        elif score < 2.8: return 'moderate'
        else: return 'high'
        
    glycemic_variability = [calc_gv_level(h, d) for h, d in zip(hba1c, disease_duration_years)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_id,
        'age': age,
        'sex': sex,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi,
        'waist_hip_ratio': waist_hip_ratio,
        'hba1c': hba1c,
        'fasting_glucose_mgdl': fasting_glucose_mgdl,
        'serum_creatinine_mgdl': serum_creatinine_mgdl,
        'egfr': egfr,
        'total_cholesterol': total_cholesterol,
        'triglycerides': triglycerides,
        'hdl': hdl,
        'ldl': ldl,
        'disease_duration_years': disease_duration_years,
        'has_hypertension': has_hypertension,
        'metformin': metformin,
        'sglt2_inhibitor': sglt2_inhibitor,
        'sulfonylurea': sulfonylurea,
        'insulin_basal': insulin_basal,
        'insulin_bolus': insulin_bolus,
        'insulin_resistance_level': insulin_resistance_level,
        'glycemic_variability': glycemic_variability
    })
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved EHR profiles to {output_path}")
    
    print("\nSummary Statistics:")
    print("-" * 30)
    print(df[['age', 'bmi', 'hba1c', 'fasting_glucose_mgdl']].describe())
    print("\nMedication Usage:")
    for med in ['metformin', 'sglt2_inhibitor', 'sulfonylurea', 'insulin_basal', 'insulin_bolus']:
        print(f"{med}: {df[med].mean()*100:.1f}%")

if __name__ == "__main__":
    generate_ehr_data()
