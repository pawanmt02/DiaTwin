import { useEffect, useState } from 'react';
import { Activity, Menu, Bell, LogOut } from 'lucide-react';
import PatientList from './components/PatientList';
import PatientCard from './components/PatientCard';
import GlucoseChart from './components/GlucoseChart';
import WhatIfSimulator from './components/WhatIfSimulator';
import AlertFeed from './components/AlertFeed';
import AuthScreen, { AuthUser } from './components/AuthScreen';
import { apiClient } from './api/client';
import { Patient, CGMData, Alert, AlertSummary } from './types';

const USER_STORAGE_KEY = 'diatwin_current_user';
const SESSION_STORAGE_KEY = 'diatwin_is_authenticated';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(SESSION_STORAGE_KEY) === 'true';
  });
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(() => {
    if (typeof window === 'undefined') return null;
    const savedUser = window.localStorage.getItem(USER_STORAGE_KEY);
    return savedUser ? JSON.parse(savedUser) : null;
  });

  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const [cgmData, setCgmData] = useState<CGMData[]>([]);
  const [alerts, setAlerts] = useState<{ alerts: Alert[], summary: AlertSummary } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isAuthenticated) return;

    const initData = async () => {
      setLoading(true);
      try {
        const [patientsData, alertsData] = await Promise.all([
          apiClient.fetchPatients(),
          apiClient.fetchAlerts()
        ]);
        setPatients(patientsData);
        setAlerts(alertsData);
        if (patientsData.length > 0) {
          setSelectedPatientId(patientsData[0].id);
        }
      } catch (err) {
        console.error("Failed to load initial data", err);
      } finally {
        setLoading(false);
      }
    };
    initData();
  }, [isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated || !selectedPatientId) {
      return;
    }

    const fetchPatientData = async () => {
      try {
        const cgm = await apiClient.fetchPatientCGM(selectedPatientId);
        setCgmData(cgm);
      } catch (err) {
        console.error("Failed to load patient CGM", err);
      }
    };
    fetchPatientData();
  }, [isAuthenticated, selectedPatientId]);

  const handleLogin = (user: AuthUser) => {
    setCurrentUser(user);
    setIsAuthenticated(true);
    window.localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));
    window.localStorage.setItem(SESSION_STORAGE_KEY, 'true');
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setIsAuthenticated(false);
    window.localStorage.removeItem(USER_STORAGE_KEY);
    window.localStorage.setItem(SESSION_STORAGE_KEY, 'false');
  };

  const selectedPatient = patients.find(p => p.id === selectedPatientId);

  if (!isAuthenticated) {
    return <AuthScreen onLogin={handleLogin} />;
  }

  return (
    <div className="flex h-screen w-full bg-slate-50 overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-80 bg-slate-900 border-r border-slate-800 flex flex-col h-full flex-shrink-0">
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-teal-500/10 p-2 rounded-lg text-teal-400">
              <Activity size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">DiaTwin</h1>
              <div className="text-[10px] uppercase tracking-wider text-teal-400 font-semibold">Clinician Portal</div>
            </div>
          </div>
          <button className="text-slate-400 hover:text-white transition-colors lg:hidden">
            <Menu size={24} />
          </button>
        </div>
        
        <div className="flex-1 overflow-hidden">
          <PatientList 
            patients={patients} 
            selectedId={selectedPatientId} 
            onSelect={setSelectedPatientId} 
          />
        </div>

        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 text-center">
          DiaTwin v1.0.0-beta<br/>
          Secure Clinical Environment
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 py-3 px-6 flex justify-between items-center flex-shrink-0">
          <h2 className="text-lg font-bold text-slate-800">Metabolic Digital Twin Dashboard</h2>
          <div className="flex items-center gap-4">
            <button className="relative p-2 text-slate-500 hover:bg-slate-100 rounded-full transition-colors">
              <Bell size={20} />
              {alerts && alerts.alerts.length > 0 && (
                <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
              )}
            </button>
            <div className="flex items-center gap-3 border-l border-slate-200 pl-4">
              <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center text-teal-700 font-bold text-sm">
                {currentUser?.name?.charAt(0)?.toUpperCase() || 'D'}
              </div>
              <div className="text-sm font-semibold text-slate-700 hidden sm:block">
                {currentUser?.name || 'Clinician'}
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
            >
              <LogOut size={16} />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </header>

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          {loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="animate-spin text-teal-600"><Activity size={48} /></div>
            </div>
          ) : selectedPatient ? (
            <div className="max-w-7xl mx-auto space-y-6">
              
              {/* Top Row: Patient Info & Alerts */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <PatientCard patient={selectedPatient} />
                </div>
                <div className="lg:col-span-1 h-full">
                  {alerts && <AlertFeed alerts={alerts.alerts} summary={alerts.summary} />}
                </div>
              </div>

              {/* Main Chart */}
              <div className="w-full">
                <GlucoseChart data={cgmData} />
              </div>

              {/* Simulator */}
              <div className="w-full">
                <WhatIfSimulator patientId={selectedPatient.id} />
              </div>

              {/* Footer */}
              <footer className="pt-8 pb-4 text-center text-sm text-slate-500">
                DiaTwin is an investigational decision-support prototype. Not for primary diagnosis.
              </footer>

            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500">
              Select a patient to view their Digital Twin.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
