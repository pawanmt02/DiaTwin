import { useMemo } from 'react';
import { CGMData } from '../types';
import { 
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer 
} from 'recharts';

interface GlucoseChartProps {
  data: CGMData[];
}

const GlucoseChart: React.FC<GlucoseChartProps> = ({ data }) => {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map(d => {
      const date = new Date(d.timestamp);
      return {
        timeLabel: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        historyValue: d.isPredicted ? undefined : d.value,
        predictedValue: d.isPredicted ? (d.q50 ?? d.value) : undefined,
        q10: d.isPredicted ? d.q10 : undefined,
        q90: d.isPredicted ? d.q90 : undefined,
      };
    });
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 w-full h-[500px] flex items-center justify-center text-slate-400">
        No glucose data available. Select a patient.
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-slate-200 shadow-md rounded-md text-sm">
          <p className="font-semibold text-slate-700 mb-1">{label}</p>
          {payload.map((entry: any, index: number) => {
            if (entry.value == null || entry.dataKey === 'q10') return null;
            const label2 = entry.dataKey === 'q90' ? '80% CI Upper' : entry.name;
            return (
              <p key={index} style={{ color: entry.color }}>
                {label2}: {Math.round(entry.value)} mg/dL
              </p>
            );
          })}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 w-full">
      <h3 className="text-lg font-bold text-slate-800 mb-4">Dynamic Glucose Forecast Canvas</h3>
      <div className="w-full h-[420px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="timeLabel" minTickGap={40} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis domain={[40, 300]} tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} />
            
            {/* Threshold reference lines */}
            <ReferenceLine y={180} stroke="#ef4444" strokeDasharray="6 4" label={{ position: 'insideTopLeft', value: '180 mg/dL', fill: '#ef4444', fontSize: 11 }} />
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="6 4" label={{ position: 'insideBottomLeft', value: '70 mg/dL', fill: '#ef4444', fontSize: 11 }} />
            
            {/* 80% Confidence Band: rendered as stacked areas, lower invisible */}
            <Area 
              type="monotone" 
              dataKey="q10" 
              fill="transparent"
              stroke="none" 
              stackId="confidence"
              legendType="none"
            />
            <Area 
              type="monotone" 
              dataKey="q90" 
              fill="#99f6e4"
              stroke="none" 
              fillOpacity={0.4} 
              stackId="confidence"
              name="80% Confidence Band" 
            />
            
            {/* Historical Data Line */}
            <Line 
              type="monotone" 
              dataKey="historyValue" 
              stroke="#0f172a" 
              strokeWidth={2.5} 
              dot={false} 
              name="Historical Glucose"
              connectNulls={false}
            />
            
            {/* Predicted Median Line */}
            <Line 
              type="monotone" 
              dataKey="predictedValue" 
              stroke="#0d9488" 
              strokeWidth={2.5} 
              strokeDasharray="5 5" 
              dot={false} 
              name="Predicted (Median)"
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default GlucoseChart;
