import { Patient, CGMData, SimulationResult, Alert, AlertSummary } from '../types';
import { mockPatients, generateMockCGM, mockAlerts, mockAlertSummary } from '../mockData';

const BASE_URL = 'http://localhost:8000/api/v1';

export const apiClient = {
  async fetchPatients(): Promise<Patient[]> {
    try {
      const response = await fetch(`${BASE_URL}/patients`);
      if (!response.ok) throw new Error('Failed to fetch');
      return await response.json();
    } catch (e) {
      console.warn('Using mock data for patients', e);
      return mockPatients;
    }
  },

  async fetchPatient(id: string): Promise<Patient> {
    try {
      const response = await fetch(`${BASE_URL}/patients/${id}`);
      if (!response.ok) throw new Error('Failed to fetch');
      return await response.json();
    } catch (e) {
      console.warn('Using mock data for patient', e);
      return mockPatients.find(p => p.id === id) || mockPatients[0];
    }
  },

  async fetchPatientCGM(id: string): Promise<CGMData[]> {
    try {
      const response = await fetch(`${BASE_URL}/patients/${id}/cgm`);
      if (!response.ok) throw new Error('Failed to fetch');
      return await response.json();
    } catch (e) {
      console.warn('Using mock data for CGM', e);
      return generateMockCGM();
    }
  },

  async predictTrajectory(id: string): Promise<CGMData[]> {
    try {
      const response = await fetch(`${BASE_URL}/predict/${id}`);
      if (!response.ok) throw new Error('Failed to fetch');
      return await response.json();
    } catch (e) {
      console.warn('Using mock data for prediction', e);
      return generateMockCGM(); // our mock cgm includes predictions
    }
  },

  async simulateIntervention(params: any): Promise<SimulationResult> {
    try {
      const response = await fetch(`${BASE_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (!response.ok) throw new Error('Failed to simulate');

      const data = await response.json();
      const baselineSeries = Array.isArray(data.glucose_baseline) ? data.glucose_baseline : [];
      const interventionSeries = Array.isArray(data.glucose_intervention) ? data.glucose_intervention : [];
      const now = Date.now();

      return {
        baseline: baselineSeries.map((value: number, index: number) => ({
          timestamp: new Date(now + index * 5 * 60000).toISOString(),
          value: Number(value),
          isPredicted: false,
        })),
        intervention: interventionSeries.map((value: number, index: number) => ({
          timestamp: new Date(now + index * 5 * 60000).toISOString(),
          value: Number(value),
          isPredicted: true,
        })),
        peakReduction: Number(data.peak_reduction_mgdl ?? 0),
        aucReduction: Number(data.auc_reduction ?? 0),
        recommendation: data.recommendation ?? 'Simulation complete.',
      };
    } catch (e) {
      console.warn('Using mock data for simulation', e);
      
      const carbs = params.carb_grams || params.carbLoad || 50;
      const exercise = params.exercise_duration_min || params.exerciseDuration || 15;
      const now = new Date();
      const baseline: CGMData[] = [];
      const intervention: CGMData[] = [];
      
      for (let i = 0; i <= 36; i++) {
        const t = i * 5; // minutes
        const ts = new Date(now.getTime() + t * 60000).toISOString();
        const mealEffect = carbs * 1.5 * Math.exp(-((t - 45) ** 2) / 800);
        const baseGlucose = 120 + mealEffect + (Math.random() - 0.5) * 4;
        const exerciseBlunt = exercise > 0 ? exercise * 0.8 * Math.exp(-((t - 30) ** 2) / 600) : 0;
        const intGlucose = baseGlucose - exerciseBlunt;
        
        baseline.push({ timestamp: ts, value: Math.round(baseGlucose), isPredicted: true });
        intervention.push({ timestamp: ts, value: Math.round(intGlucose), isPredicted: true });
      }

      const peakBase = Math.max(...baseline.map(d => d.value));
      const peakInt = Math.max(...intervention.map(d => d.value));

      return {
        baseline,
        intervention,
        peakReduction: Math.max(0, peakBase - peakInt),
        aucReduction: Math.round((peakBase - peakInt) * 30),
        recommendation: exercise > 0 
          ? `A ${exercise}-minute walk post-meal reduces the predicted peak by ${Math.round(peakBase - peakInt)} mg/dL. This intervention is clinically meaningful.`
          : "No exercise intervention applied. Consider a 15-30 minute post-meal walk to blunt the glucose spike."
      };
    }
  },

  async fetchAlerts(): Promise<{alerts: Alert[], summary: AlertSummary}> {
    try {
      const response = await fetch(`${BASE_URL}/alerts`);
      if (!response.ok) throw new Error('Failed to fetch');

      const data = await response.json();
      const rawAlerts = Array.isArray(data.alerts) ? data.alerts : [];

      return {
        alerts: rawAlerts.map((alert: any, index: number) => ({
          id: `${alert.patient_id ?? alert.patientId ?? 'alert'}-${index}`,
          patientId: alert.patient_id ?? alert.patientId ?? 'Unknown',
          type: alert.alert_type === 'hypoglycemia' ? 'hypoglycemia' : 'hyperglycemia',
          severity: alert.severity ?? 'info',
          predictedValue: Number(alert.predicted_value ?? alert.predictedValue ?? 0),
          timeUntilEventMinutes: Number(alert.predicted_time_min ?? alert.timeUntilEventMinutes ?? 0),
          message: alert.message ?? 'No message available',
        })),
        summary: {
          totalPatients: Number(data.total_patients ?? data.totalPatients ?? rawAlerts.length ?? 0),
          stable: Number(data.patients_stable ?? data.stable ?? 0),
          atRisk: Number(data.patients_at_risk ?? data.atRisk ?? 0),
        },
      };
    } catch (e) {
      console.warn('Using mock data for alerts', e);
      return { alerts: mockAlerts, summary: mockAlertSummary };
    }
  }
};
