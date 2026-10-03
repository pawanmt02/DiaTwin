import React, { useState } from 'react';
import { Patient } from '../types';
import { Search, User } from 'lucide-react';

interface PatientListProps {
  patients: Patient[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const PatientList: React.FC<PatientListProps> = ({ patients, selectedId, onSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPatients = patients.filter(p => 
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getRiskDot = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-500';
      case 'moderate': return 'bg-yellow-500';
      case 'high': return 'bg-orange-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="flex flex-col h-full text-slate-300">
      <div className="p-4">
        <div className="relative">
          <input
            type="text"
            placeholder="Search patients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-800 text-white placeholder-slate-400 border border-slate-700 rounded-lg py-2 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent text-sm"
          />
          <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto">
        <div className="px-3 pb-4 space-y-1">
          {filteredPatients.map(patient => (
            <button
              key={patient.id}
              onClick={() => onSelect(patient.id)}
              className={`w-full text-left p-3 rounded-lg transition-colors flex items-center justify-between ${
                selectedId === patient.id 
                  ? 'bg-slate-800 border border-slate-700 shadow-sm' 
                  : 'hover:bg-slate-800 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="bg-slate-700 p-2 rounded-full">
                  <User size={16} className="text-slate-300" />
                </div>
                <div>
                  <div className={`font-semibold text-sm ${selectedId === patient.id ? 'text-white' : 'text-slate-200'}`}>
                    {patient.name}
                  </div>
                  <div className="text-xs text-slate-400 font-mono mt-0.5">{patient.id}</div>
                </div>
              </div>
              
              <div className="flex flex-col items-end gap-1">
                <span className="text-xs font-semibold text-slate-300 bg-slate-700 px-2 py-0.5 rounded">
                  A1c {patient.hba1c}
                </span>
                <div className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${getRiskDot(patient.riskLevel)}`}></div>
                  <span className="text-[10px] uppercase font-bold text-slate-500">{patient.riskLevel}</span>
                </div>
              </div>
            </button>
          ))}
          {filteredPatients.length === 0 && (
            <div className="text-center text-slate-500 py-6 text-sm">
              No patients found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatientList;
