import { Patient, CGMData, Alert, AlertSummary } from './types';

export const mockPatients: Patient[] = [
  {
    id: "IND-0042",
    name: "Rajesh Sharma",
    age: 54,
    sex: "Male",
    bmi: 27.4,
    hba1c: 8.4,
    egfr: 68,
    medications: ["Metformin 1000mg", "SGLT2 Inhibitor"],
    insulinResistanceLevel: "High",
    riskLevel: "high"
  },
  {
    id: "IND-0078",
    name: "Priya Patel",
    age: 48,
    sex: "Female",
    bmi: 25.1,
    hba1c: 7.2,
    egfr: 92,
    medications: ["Metformin 500mg"],
    insulinResistanceLevel: "Moderate",
    riskLevel: "moderate"
  },
  {
    id: "IND-0135",
    name: "Arun Krishnan",
    age: 61,
    sex: "Male",
    bmi: 31.2,
    hba1c: 10.1,
    egfr: 45,
    medications: ["Metformin 1000mg", "Sulfonylurea", "Basal Insulin"],
    insulinResistanceLevel: "Critical",
    riskLevel: "critical"
  },
  {
    id: "IND-0201",
    name: "Meena Gupta",
    age: 45,
    sex: "Female",
    bmi: 23.8,
    hba1c: 6.8,
    egfr: 105,
    medications: ["Metformin 500mg"],
    insulinResistanceLevel: "Low",
    riskLevel: "low"
  },
  {
    id: "IND-0312",
    name: "Vikram Singh",
    age: 58,
    sex: "Male",
    bmi: 29.8,
    hba1c: 9.3,
    egfr: 55,
    medications: ["Metformin 1000mg", "SGLT2 Inhibitor", "Sulfonylurea"],
    insulinResistanceLevel: "High",
    riskLevel: "critical"
  },
  {
    id: "IND-0467",
    name: "Sunita Reddy",
    age: 52,
    sex: "Female",
    bmi: 26.5,
    hba1c: 7.9,
    egfr: 78,
    medications: ["Metformin 1000mg", "SGLT2 Inhibitor"],
    insulinResistanceLevel: "Moderate",
    riskLevel: "moderate"
  }
];

export const generateMockCGM = (): CGMData[] => {
  const data: CGMData[] = [];
  const now = new Date();
  
  // Historical data (past 2 hours) — simulate postprandial response
  let glucose = 115; // fasting baseline
  for (let i = -120; i <= 0; i += 5) {
    const time = new Date(now.getTime() + i * 60000);
    
    // Simulate a meal at -90 minutes with postprandial rise
    const mealOffset = i + 90;
    if (mealOffset >= 0 && mealOffset < 60) {
      glucose += 3.5 * Math.exp(-mealOffset / 25); // rising phase
    } else if (mealOffset >= 60) {
      glucose -= 1.8 * Math.exp(-(mealOffset - 60) / 30); // falling phase
    }
    
    glucose += (Math.random() - 0.5) * 4; // sensor noise
    glucose = Math.max(70, Math.min(glucose, 280));
    
    data.push({
      timestamp: time.toISOString(),
      value: Math.round(glucose),
      isPredicted: false
    });
  }
  
  // Predicted data (next 2 hours) — forecasted spike
  const lastValue = data[data.length - 1].value;
  for (let i = 5; i <= 120; i += 5) {
    const time = new Date(now.getTime() + i * 60000);
    
    // Simulate continued postprandial descent then stabilization
    const trend = -15 * Math.exp(-i / 40) + 8 * Math.sin(i / 30);
    const q50 = Math.round(lastValue + trend + (Math.random() - 0.5) * 3);
    const uncertainty = 8 + i * 0.3; // grows with horizon
    
    data.push({
      timestamp: time.toISOString(),
      value: q50,
      isPredicted: true,
      q50: q50,
      q10: Math.round(q50 - uncertainty),
      q90: Math.round(q50 + uncertainty)
    });
  }
  
  return data;
};

export const mockAlerts: Alert[] = [
  {
    id: "A-1",
    patientId: "IND-0135",
    type: "hyperglycemia",
    severity: "critical",
    predictedValue: 248,
    timeUntilEventMinutes: 25,
    message: "Predicted severe spike to 248 mg/dL at t+25min"
  },
  {
    id: "A-2",
    patientId: "IND-0312",
    type: "hyperglycemia",
    severity: "warning",
    predictedValue: 195,
    timeUntilEventMinutes: 40,
    message: "Predicted hyperglycemia (195 mg/dL) at t+40min"
  },
  {
    id: "A-3",
    patientId: "IND-0042",
    type: "rapid_rise",
    severity: "warning",
    predictedValue: 188,
    timeUntilEventMinutes: 45,
    message: "Rapid postprandial rise detected, spike predicted at t+45min"
  },
  {
    id: "A-4",
    patientId: "IND-0078",
    type: "hypoglycemia",
    severity: "warning",
    predictedValue: 65,
    timeUntilEventMinutes: 60,
    message: "Nocturnal hypoglycemia risk (65 mg/dL) at t+60min"
  }
];

export const mockAlertSummary: AlertSummary = {
  totalPatients: 1000,
  stable: 847,
  atRisk: 153
};
