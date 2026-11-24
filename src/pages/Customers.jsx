import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import PageHeader from '../components/PageHeader';
import { SectionCard, StatCard, RiskBar } from '../components/ui';
import { customerTrend, customerStats } from '../data/mockData';
import { predictChurn } from '../api';

const DEFAULT_CUSTOMER = {
  days_since_last_visit: 45,
  visits_last_90d: 2,
  avg_basket_value: 1840,
  loyalty_member: 0,
  tenure_months: 8,
};

const FIELDS = [
  { key: 'days_since_last_visit', label: 'Days since last visit', hint: '0 to 120', step: 1 },
  { key: 'visits_last_90d', label: 'Visits in last 90 days', hint: '0 to 20', step: 1 },
  { key: 'avg_basket_value', label: 'Avg. basket value (LKR)', hint: 'Training range 500 to 5000', step: 100 },
  { key: 'tenure_months', label: 'Months as a customer', hint: '1 to 48', step: 1 },
];

/**
 * Customers (route: "/customers")
 * Shows new vs. returning customers plus a churn-risk simulator that calls
 * the trained churn model (/predict-churn) on the Flask backend.
 */
export default function Customers() {
  const [inputs, setInputs] = useState(DEFAULT_CUSTOMER);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runPrediction = (values) => {
    setLoading(true);
    setError('');
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));
    predictChurn(payload)
      .then((res) => setResult(res))
      .catch((err) => {
        console.error('Churn prediction failed:', err);
        setResult(null);
        setError(err.message || 'Could not reach the backend. Is "python app.py" running on port 5000?');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    runPrediction(DEFAULT_CUSTOMER);
  }, []);

  const handleChange = (key, value) => setInputs((prev) => ({ ...prev, [key]: value }));
  const handleReset = () => {
    setInputs(DEFAULT_CUSTOMER);
    runPrediction(DEFAULT_CUSTOMER);
  };

  const probPct = result ? Math.round(result.churn_probability * 100) : 0;

  return (
    <>
      <PageHeader caseRef="Exhibit · Customers" title="Customers" subtitle="Behaviour patterns and retention analysis" />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        {customerStats.map((s, i) => (
          <div key={s.label} className="rise-in" style={{ '--delay': `${i * 60}ms` }}>
            <StatCard {...s} />
          </div>
        ))}
      </div>

      <SectionCard
        eyebrow="Model test"
        title="Churn risk simulator"
        subtitle="Describe one customer and the trained model estimates the chance they stop coming back"
        className="rise-in mb-4"
        action={
          <span className={`font-mono text-[10.5px] tracking-[0.1em] uppercase ${result ? 'text-vital' : 'text-pulse'}`}>
            {result ? '● Backend connected' : '● Backend offline'}
          </span>
        }
      >
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
          {FIELDS.map((f) => (
            <label key={f.key} className="block">
              <span className="font-mono text-[10px] tracking-[0.1em] uppercase text-ink-faint">{f.label}</span>
              <input
                type="number"
                className="input mt-1"
                value={inputs[f.key]}
                step={f.step}
                onChange={(e) => handleChange(f.key, e.target.value)}
              />
              <span className="text-[11px] text-ink-faint">{f.hint}</span>
            </label>
          ))}
          <label className="block">
            <span className="font-mono text-[10px] tracking-[0.1em] uppercase text-ink-faint">Loyalty member</span>
            <select
              className="input mt-1"
              value={inputs.loyalty_member}
              onChange={(e) => handleChange('loyalty_member', e.target.value)}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
            <span className="text-[11px] text-ink-faint">Enrolled in loyalty program</span>
          </label>
        </div>

        <div className="flex items-center gap-3 mt-4 flex-wrap">
          <button
            type="button"
            onClick={() => runPrediction(inputs)}
            disabled={loading}
            className="inline-flex items-center justify-center bg-vital text-white text-[13px] font-medium px-4 py-2 rounded-md hover:bg-vital-deep active:scale-[0.97] transition-all disabled:opacity-60"
          >
            {loading ? 'Running…' : 'Predict churn risk'}
          </button>
          <button type="button" onClick={handleReset} className="text-[13px] text-ink-soft hover:text-ink px-3 py-2">
            Reset
          </button>
          {error && <span className="text-[12px] text-pulse">{error}</span>}
        </div>

        {result && (
          <div className="mt-5 border-t border-line pt-4 max-w-xl">
            <RiskBar label="Churn probability (next 90 days)" level={result.risk_level} value={probPct} />
            <p className="text-[12px] text-ink-faint mt-2 font-mono">
              Model: {result.model.replace('_', ' ')} · exact value {(result.churn_probability * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </SectionCard>

      <SectionCard eyebrow="Exhibit A" title="New vs returning customers" subtitle="Monthly transaction count" className="rise-in mb-4" style={{ '--delay': '220ms' }}>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={customerTrend} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#233047" vertical={false} />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: '#8493AD' }} />
              <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#8493AD' }} />
              <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #233047', background: '#131C2E', color: '#E9EEF9', fontFamily: 'IBM Plex Mono, monospace' }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="returning" name="Returning" stackId="a" fill="#3E7BFA" radius={[0, 0, 0, 0]} />
              <Bar dataKey="new" name="New" stackId="a" fill="#E0A83E" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <SectionCard eyebrow="Analyst note" title="Insight" className="rise-in" style={{ '--delay': '280ms' }}>
        <p className="text-[13px] text-ink-soft leading-relaxed">
          Returning customers have declined steadily since January while new customer acquisition has stayed flat.
          With no loyalty program active, first-time buyers have no incentive to return, which lines up with the
          "no customer retention strategy" root cause in the latest diagnostic report.
        </p>
      </SectionCard>
    </>
  );
}
