import React from 'react';
import { Alert, AlertSummary } from '../types';
import { AlertCircle, AlertTriangle, Info, Clock } from 'lucide-react';

interface AlertFeedProps {
  alerts: Alert[];
  summary: AlertSummary;
}

const AlertFeed: React.FC<AlertFeedProps> = ({ alerts, summary }) => {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertCircle className="text-red-500" size={24} />;
      case 'warning': return <AlertTriangle className="text-amber-500" size={24} />;
      default: return <Info className="text-blue-500" size={24} />;
    }
  };

  const getSeverityClasses = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-50 border-red-200';
      case 'warning': return 'bg-amber-50 border-amber-200';
      default: return 'bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-6">
        <AlertTriangle className="text-orange-500" size={24} />
        <h3 className="text-xl font-bold text-slate-800">Automated Early Warning Feed</h3>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-6 bg-slate-50 p-3 rounded-lg border border-slate-100">
        <div className="text-center">
          <div className="text-xs text-slate-500 font-semibold uppercase">Total</div>
          <div className="text-lg font-bold text-slate-700">{summary.totalPatients}</div>
        </div>
        <div className="text-center border-l border-r border-slate-200">
          <div className="text-xs text-slate-500 font-semibold uppercase">Stable</div>
          <div className="text-lg font-bold text-emerald-600">{summary.stable}</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-slate-500 font-semibold uppercase">At Risk</div>
          <div className="text-lg font-bold text-orange-600">{summary.atRisk}</div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 space-y-3">
        {alerts.map((alert) => (
          <div key={alert.id} className={`p-4 rounded-lg border ${getSeverityClasses(alert.severity)}`}>
            <div className="flex items-start gap-3">
              <div className="mt-1">
                {getSeverityIcon(alert.severity)}
              </div>
              <div className="flex-1">
                <div className="flex justify-between items-start">
                  <span className="font-bold text-slate-800">{alert.patientId}</span>
                  <span className="flex items-center gap-1 text-xs font-semibold text-slate-600 bg-white px-2 py-1 rounded-full shadow-sm">
                    <Clock size={12} /> {alert.timeUntilEventMinutes} min
                  </span>
                </div>
                <p className="text-sm text-slate-700 mt-1 font-medium">{alert.message}</p>
                <div className="mt-2 text-xs font-semibold">
                  Predicted Value: <span className={alert.severity === 'critical' ? 'text-red-600' : 'text-amber-600'}>
                    {alert.predictedValue} mg/dL
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
        {alerts.length === 0 && (
          <div className="text-center text-slate-500 py-8">
            No active alerts
          </div>
        )}
      </div>
    </div>
  );
};

export default AlertFeed;
