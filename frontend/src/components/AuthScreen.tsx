import { useMemo, useState } from 'react';
import { Activity, ArrowRight, CheckCircle2, Shield, UserPlus } from 'lucide-react';

export type AuthMode = 'login' | 'signup';

export interface AuthUser {
  name: string;
  email: string;
  role: string;
}

interface AuthScreenProps {
  onLogin: (user: AuthUser) => void;
}

const STORAGE_KEY = 'diatwin_accounts';

const getStoredAccounts = (): Array<{ name: string; email: string; password: string; role: string }> => {
  if (typeof window === 'undefined') return [];

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
};

const seedAccounts = [
  { name: 'Dr. Clinician', email: 'doctor@diatwin.com', password: 'admin123', role: 'Clinician' },
];

const AuthScreen: React.FC<AuthScreenProps> = ({ onLogin }) => {
  const [mode, setMode] = useState<AuthMode>('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('doctor@diatwin.com');
  const [password, setPassword] = useState('admin123');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const featureList = useMemo(
    () => [
      'Secure clinician access',
      'Patient risk monitoring',
      'Real-time glucose forecasting',
    ],
    [],
  );

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setError('');

    if (!email.trim() || !password.trim()) {
      setError('Please enter both email and password.');
      return;
    }

    if (mode === 'signup') {
      if (!name.trim()) {
        setError('Please enter your full name.');
        return;
      }

      if (password !== confirmPassword) {
        setError('Passwords do not match.');
        return;
      }
    }

    const existingAccounts = getStoredAccounts();
    const accountList = existingAccounts.length > 0 ? existingAccounts : seedAccounts;

    if (mode === 'login') {
      const account = accountList.find(
        (item) => item.email.toLowerCase() === email.toLowerCase() && item.password === password,
      );

      if (!account) {
        setError('Invalid email or password. Try doctor@diatwin.com / admin123');
        return;
      }

      onLogin({ name: account.name, email: account.email, role: account.role });
      return;
    }

    const duplicate = accountList.some((item) => item.email.toLowerCase() === email.toLowerCase());
    if (duplicate) {
      setError('An account with this email already exists.');
      return;
    }

    const nextAccounts = [...accountList, { name, email, password, role: 'Clinician' }];
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextAccounts));
    onLogin({ name, email, role: 'Clinician' });
  };

  const toggleMode = () => {
    setMode((current) => (current === 'login' ? 'signup' : 'login'));
    setError('');
  };

  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl grid overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-2xl lg:grid-cols-[1.1fr_0.9fr]">
        <div className="bg-slate-900 p-8 text-white md:p-12">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-teal-500/10 p-2 text-teal-300">
              <Activity size={28} />
            </div>
            <div>
              <div className="text-xs uppercase tracking-[0.28em] text-teal-300">DiaTwin</div>
              <div className="text-2xl font-bold">Clinical Portal</div>
            </div>
          </div>

          <div className="mt-12 space-y-6">
            <div className="text-4xl font-bold leading-tight">
              Strengthen patient insight with predictive glucose intelligence.
            </div>
            <p className="max-w-md text-slate-300">
              Monitor early warning signals, model counterfactual interventions, and support safer clinical decisions.
            </p>

            <div className="space-y-4 pt-4">
              {featureList.map((item) => (
                <div key={item} className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-800/60 p-3">
                  <CheckCircle2 className="text-teal-300" size={18} />
                  <span className="text-slate-100">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="p-8 md:p-12">
          <div className="mb-8 flex items-center gap-2 text-slate-600">
            <Shield size={18} className="text-teal-600" />
            <span className="text-sm font-semibold uppercase tracking-[0.2em]">
              {mode === 'login' ? 'Welcome back' : 'Create account'}
            </span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {mode === 'signup' && (
              <div>
                <label htmlFor="name" className="mb-2 block text-sm font-semibold text-slate-700">
                  Full name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Dr. Maya Patel"
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-slate-800 outline-none transition focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-200"
                />
              </div>
            )}

            <div>
              <label htmlFor="email" className="mb-2 block text-sm font-semibold text-slate-700">
                Email address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="doctor@diatwin.com"
                className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-slate-800 outline-none transition focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-200"
              />
            </div>

            <div>
              <label htmlFor="password" className="mb-2 block text-sm font-semibold text-slate-700">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter password"
                className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-slate-800 outline-none transition focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-200"
              />
            </div>

            {mode === 'signup' && (
              <div>
                <label htmlFor="confirmPassword" className="mb-2 block text-sm font-semibold text-slate-700">
                  Confirm password
                </label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  placeholder="Re-enter password"
                  className="w-full rounded-xl border border-slate-300 bg-slate-50 px-4 py-3 text-slate-800 outline-none transition focus:border-teal-500 focus:bg-white focus:ring-2 focus:ring-teal-200"
                />
              </div>
            )}

            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <button
              type="submit"
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-teal-600 px-4 py-3 text-base font-semibold text-white transition hover:bg-teal-700"
            >
              {mode === 'login' ? 'Login to dashboard' : 'Create account'}
              <ArrowRight size={18} />
            </button>

            <button
              type="button"
              onClick={toggleMode}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              <UserPlus size={16} />
              {mode === 'login' ? 'Create a new account' : 'Already have an account? Login'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AuthScreen;
