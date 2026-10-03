export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: 'Male' | 'Female' | 'Other';
  bmi: number;
  hba1c: number;
  egfr: number;
  medications: string[];
  insulinResistanceLevel: 'Low' | 'Moderate' | 'High' | 'Critical';
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
}

export interface CGMData {
  timestamp: string; // ISO string
  value: number; // glucose in mg/dL
  isPredicted: boolean;
  q10?: number;
  q50?: number;
  q90?: number;
}

export interface SimulationResult {
  baseline: CGMData[];
  intervention: CGMData[];
  peakReduction: number;
  aucReduction: number;
  recommendation: string;
}

export interface Alert {
  id: string;
  patientId: string;
  type: 'hypoglycemia' | 'hyperglycemia' | 'rapid_drop' | 'rapid_rise';
  severity: 'info' | 'warning' | 'critical';
  predictedValue: number;
  timeUntilEventMinutes: number;
  message: string;
}

export interface AlertSummary {
  totalPatients: number;
  stable: number;
  atRisk: number;
}
