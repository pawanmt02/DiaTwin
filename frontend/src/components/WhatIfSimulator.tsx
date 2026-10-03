import { useState } from 'react';
import { Activity, Play, FlaskConical, UtensilsCrossed } from 'lucide-react';
import { apiClient } from '../api/client';
import { SimulationResult } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

interface WhatIfSimulatorProps {
  patientId: string;
}

const WhatIfSimulator: React.FC<WhatIfSimulatorProps> = ({ patientId }) => {
  const [carbLoad, setCarbLoad] = useState(45);
  const [exerciseDuration, setExerciseDuration] = useState(20);
  const [exerciseIntensity, setExerciseIntensity] = useState('Moderate');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);

  const handleSimulate = async () => {
    setLoading(true);
    try {
      const intensityMap: Record<string, number> = { Light: 0.5, Moderate: 1.0, Vigorous: 1.8 };
      const res = await apiClient.simulateIntervention({
        patient_id: patientId,
        current_glucose: 130,
        carb_grams: carbLoad,
        exercise_duration_min: exerciseDuration,
        exercise_intensity: intensityMap[exerciseIntensity] ?? 1.0,
        simulation_duration_min: 180,
      });
      setResult(res);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.baseline.map((b, i) => {
    const int = result.intervention[i];
    return {
      time: `${i * 5}m`,
      baseline: Math.round(b.value),
      intervention: int ? Math.round(int.value) : undefined,
    };
  });

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
      <div className="flex items-center gap-2 mb-6 border-b border-slate-100 pb-4">
        <FlaskConical className="text-teal-600" size={24} />
        <h3 className="text-xl font-bold text-slate-800">Counterfactual "What-If" Simulator</h3>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="space-y-6 lg:col-span-1">
          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-2">
              <UtensilsCrossed size={16} className="text-orange-500" />
              Carbohydrate Load: <span className="text-teal-600">{carbLoad}g</span>
            </label>
            <input 
              type="range" 
              min="0" max="150" 
              value={carbLoad} 
              onChange={(e) => setCarbLoad(parseInt(e.target.value))}
              className="w-full accent-teal-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>0g</span><span>75g</span><span>150g</span>
            </div>
          </div>

          <div>
            <label className="flex items-center gap-2 text-sm font-semibold text-slate-700 mb-2">
              <Activity size={16} className="text-blue-500" />
              Exercise Duration: <span className="text-teal-600">{exerciseDuration} min</span>
            </label>
            <input 
              type="range" 
              min="0" max="60" step="5"
              value={exerciseDuration} 
              onChange={(e) => setExerciseDuration(parseInt(e.target.value))}
              className="w-full accent-teal-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>0 min</span><span>30 min</span><span>60 min</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-semibold text-slate-700 mb-2 block">
              Exercise Intensity
            </label>
            <select 
              value={exerciseIntensity}
              onChange={(e) => setExerciseIntensity(e.target.value)}
              className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:outline-none bg-white"
            >
              <option>Light</option>
              <option>Moderate</option>
              <option>Vigorous</option>
            </select>
          </div>

          <button 
            onClick={handleSimulate}
            disabled={loading}
            className="w-full bg-teal-600 hover:bg-teal-700 text-white font-semibold py-3 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 cursor-pointer"
          >
            {loading ? (
              <span className="animate-pulse">Simulating...</span>
            ) : (
              <>
                <Play size={18} /> Run In-Silico Simulation
              </>
            )}
          </button>
        </div>

        <div className="lg:col-span-2 bg-slate-50 p-4 rounded-xl border border-slate-200">
          {result ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-white p-3 rounded-lg shadow-sm border border-slate-100">
                  <div className="text-slate-500 text-xs uppercase font-semibold">Peak Reduction</div>
                  <div className="text-2xl font-bold text-teal-600">-{result.peakReduction.toFixed(1)} <span className="text-sm">mg/dL</span></div>
                </div>
                <div className="bg-white p-3 rounded-lg shadow-sm border border-slate-100">
                  <div className="text-slate-500 text-xs uppercase font-semibold">AUC Reduction</div>
                  <div className="text-2xl font-bold text-teal-600">-{result.aucReduction.toFixed(0)} <span className="text-sm">mg·min/dL</span></div>
                </div>
              </div>
              <div className="bg-teal-50 text-teal-800 p-3 rounded-lg text-sm font-medium border border-teal-100">
                {result.recommendation}
              </div>
              <div className="h-[250px] mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} minTickGap={30} />
                    <YAxis tick={{ fontSize: 11 }} domain={['dataMin - 20', 'dataMax + 20']} />
                    <Tooltip />
                    <Legend />
                    <ReferenceLine y={180} stroke="#ef4444" strokeDasharray="3 3" />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="baseline" stroke="#94a3b8" strokeWidth={2} name="Baseline (no exercise)" strokeDasharray="5 5" dot={false} />
                    <Line type="monotone" dataKey="intervention" stroke="#0d9488" strokeWidth={2.5} name="With Intervention" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 min-h-[300px]">
              <FlaskConical size={48} className="mb-4 opacity-50" />
              <p className="text-center">Adjust meal and exercise parameters,<br/>then run simulation to see projected outcomes.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WhatIfSimulator;
