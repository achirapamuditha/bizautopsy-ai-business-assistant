import React, { useEffect, useState } from 'react';
import { Download, Share2, AlertTriangle } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import HealthGauge from '../components/HealthGauge';
import { SectionCard, RiskBar, PriorityBadge } from '../components/ui';
import { business, riskAssessment, rootCauses, recommendations } from '../data/mockData';
import { predictHealth } from '../api';

// Starting values loosely based on the Silva Grocery Shop example.
const DEFAULT_PROFILE = {
  revenue_growth_pct: 3.1,
  expense_ratio_pct: 78,
  repeat_customer_rate_pct: 51,
  rent_share_pct: 28,
  inventory_days: 70,
  loyalty_active: 0,
};

const FIELDS = [
  { key: 'revenue_growth_pct', label: 'Revenue growth (%)', hint: '-20 to 25', step: 0.5 },
  { key: 'expense_ratio_pct', label: 'Expenses / revenue (%)', hint: '55 to 115', step: 1 },
  { key: 'repeat_customer_rate_pct', label: 'Repeat customer rate (%)', hint: '20 to 80', step: 1 },
  { key: 'rent_share_pct', label: 'Rent & utilities share (%)', hint: '5 to 40', step: 1 },
  { key: 'inventory_days', label: 'Inventory days on hand', hint: '10 to 120', step: 1 },
];

// Which colour the verdict stamp uses for each risk label.
const STAMP_COLOUR = {
  'High risk': 'text-pulse border-pulse/60',
  'Moderate risk': 'text-amber border-amber/60',
  'Low risk': 'text-vital border-vital/60',
};

/**
 * Diagnostics (route: "/diagnostics")
 * The verdict card at the top uses the trained health-score model
 * (/predict-health) when the backend is running, and falls back to the
 * mock score otherwise. Change the simulator inputs to see it update.
 */
export default function Diagnostics() {
  const [inputs, setInputs] = useState(DEFAULT_PROFILE);
  const [live, setLive] = useState(null); // { health_score, risk_level, model } or null
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runPrediction = (values) => {
    setLoading(true);
    setError('');
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));
    predictHealth(payload)
      .then((res) => setLive(res))
      .catch((err) => {
        console.error('Health prediction failed, showing mock score:', err);
        setLive(null);
        setError(err.message || 'Could not reach the backend. Is "python app.py" running on port 5000?');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    runPrediction(DEFAULT_PROFILE);
  }, []);

  const handleChange = (key, value) => setInputs((prev) => ({ ...prev, [key]: value }));
  const handleReset = () => {
    setInputs(DEFAULT_PROFILE);
    runPrediction(DEFAULT_PROFILE);
  };

  const score = live ? Math.round(live.health_score) : business.healthScore;
  const riskLevel = live ? live.risk_level : business.riskLevel;
  const stampColour = STAMP_COLOUR[riskLevel] || STAMP_COLOUR['Moderate risk'];

  return (
    <>
      <PageHeader
        caseRef={`Case no. ${business.analysisDate.split(' ').reverse().join('-').toUpperCase()}`}
        title="Diagnostic report"
        subtitle={`${business.name} · Analysis date: ${business.analysisDate}`}
        actions={
          <>
            <button className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-soft border border-line px-3 py-2 rounded-md hover:bg-white/5 hover:text-ink transition-colors">
              <Download size={14} />
              Export PDF
            </button>
            <button className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-soft border border-line px-3 py-2 rounded-md hover:bg-white/5 hover:text-ink transition-colors">
              <Share2 size={14} />
              Share report
            </button>
          </>
        }
      />

      <SectionCard eyebrow="Verdict" className="rise-in mb-4">
        <div className="flex flex-col md:flex-row md:items-center gap-6">
          <HealthGauge key={score} score={score} size={104} />
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1.5 flex-wrap">
              <h2 className="font-display font-semibold text-lg text-ink">Overall business health: {riskLevel}</h2>
              <span
                className={`stamp-in font-mono text-[10.5px] font-semibold uppercase tracking-[0.08em] border px-2 py-0.5 rounded-sm ${stampColour}`}
                style={{ '--stamp-rotate': '-2deg', transform: 'rotate(-2deg)' }}
              >
                {riskLevel}
              </span>
            </div>
            <p className="text-[13px] text-ink-soft">
              {business.issuesIdentified} issues identified · {business.recommendationsGenerated} recommendations
              generated · Analysis confidence: {business.analysisConfidence}%
            </p>
            <p className="font-mono text-[11px] text-ink-faint mt-1.5">
              {live ? `Score ${live.health_score} from the trained model (${live.model.replace('_', ' ')})` : 'Showing sample score (backend offline)'}
            </p>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        eyebrow="Model test"
        title="Health score simulator"
        subtitle="Change the business ratios and the trained model recalculates the score above"
        className="rise-in mb-4"
        action={
          <span className={`font-mono text-[10.5px] tracking-[0.1em] uppercase ${live ? 'text-vital' : 'text-pulse'}`}>
            {live ? '● Backend connected' : '● Backend offline'}
          </span>
        }
      >
        <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
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
            <span className="font-mono text-[10px] tracking-[0.1em] uppercase text-ink-faint">Loyalty program</span>
            <select
              className="input mt-1"
              value={inputs.loyalty_active}
              onChange={(e) => handleChange('loyalty_active', e.target.value)}
            >
              <option value={0}>Not active</option>
              <option value={1}>Active</option>
            </select>
            <span className="text-[11px] text-ink-faint">Is one running?</span>
          </label>
        </div>

        <div className="flex items-center gap-3 mt-4 flex-wrap">
          <button
            type="button"
            onClick={() => runPrediction(inputs)}
            disabled={loading}
            className="inline-flex items-center justify-center bg-vital text-white text-[13px] font-medium px-4 py-2 rounded-md hover:bg-vital-deep active:scale-[0.97] transition-all disabled:opacity-60"
          >
            {loading ? 'Running…' : 'Calculate health score'}
          </button>
          <button type="button" onClick={handleReset} className="text-[13px] text-ink-soft hover:text-ink px-3 py-2">
            Reset
          </button>
          {error && <span className="text-[12px] text-pulse">{error}</span>}
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <SectionCard eyebrow="Exhibit A" title="Risk assessment" className="rise-in" style={{ '--delay': '80ms' }}>
          <div className="flex flex-col gap-4">
            {riskAssessment.map((r) => (
              <RiskBar key={r.label} {...r} />
            ))}
          </div>
        </SectionCard>

        <SectionCard eyebrow="Exhibit B" title="Root causes identified" className="rise-in" style={{ '--delay': '140ms' }}>
          <div className="flex flex-col gap-4">
            {rootCauses.map((c) => (
              <div key={c.title} className="flex gap-2.5">
                <AlertTriangle size={15} className="text-pulse shrink-0 mt-0.5" />
                <div>
                  <p className="text-[13px] font-medium text-ink">{c.title}</p>
                  <p className="text-[12px] text-ink-soft mt-0.5">{c.description}</p>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      </div>

      <SectionCard eyebrow="Exhibit C" title="AI-generated recommendations" className="rise-in" style={{ '--delay': '200ms' }}>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {recommendations.map((rec) => (
            <div key={rec.title} className="border border-line rounded-md p-4 bg-canvas/40">
              <PriorityBadge priority={rec.priority} />
              <p className="text-[14px] font-medium text-ink mt-3">{rec.title}</p>
              <p className="text-[13px] text-ink-soft mt-1 leading-relaxed">{rec.description}</p>
            </div>
          ))}
        </div>
      </SectionCard>
    </>
  );
}
