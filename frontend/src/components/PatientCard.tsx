import { Patient } from '../types';
import { Pill } from 'lucide-react';

interface PatientCardProps {
  patient: Patient;
}

const PatientCard: React.FC<PatientCardProps> = ({ patient }) => {
  const getBadgeColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      case 'moderate': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">{patient.name}</h2>
          <p className="text-slate-500 font-medium">{patient.id} • {patient.age} y/o • {patient.sex}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getBadgeColor(patient.riskLevel)}`}>
          {patient.riskLevel.toUpperCase()} RISK
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div className="text-slate-500 text-sm mb-1">BMI</div>
          <div className="text-xl font-bold text-slate-700">{patient.bmi}</div>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div className="text-slate-500 text-sm mb-1">HbA1c</div>
          <div className="text-xl font-bold text-slate-700">{patient.hba1c}%</div>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div className="text-slate-500 text-sm mb-1">eGFR</div>
          <div className="text-xl font-bold text-slate-700">{patient.egfr}</div>
        </div>
        <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
          <div className="text-slate-500 text-sm mb-1">Insulin Res.</div>
          <div className={`text-lg font-bold ${
            patient.insulinResistanceLevel === 'High' || patient.insulinResistanceLevel === 'Critical' ? 'text-red-600' : 'text-slate-700'
          }`}>{patient.insulinResistanceLevel}</div>
        </div>
      </div>

      <div className="border-t border-slate-100 pt-4">
        <h3 className="text-sm font-semibold text-slate-500 mb-2 flex items-center gap-2">
          <Pill size={16} /> Current Medications
        </h3>
        <div className="flex flex-wrap gap-2">
          {patient.medications.map((med, idx) => (
            <span key={idx} className="bg-blue-50 text-blue-700 px-3 py-1 rounded-md text-sm font-medium border border-blue-100">
              {med}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PatientCard;
