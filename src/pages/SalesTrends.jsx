import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import PageHeader from '../components/PageHeader';
import { SectionCard, StatCard } from '../components/ui';
import { salesTrend as mockSalesTrend } from '../data/mockData';
import { predictSeries } from '../api';

const DEFAULT_INPUTS = {
  months_ahead: 6,
  start_month_index: 25,
  last_known_sales: 560000,
  marketing_spend: 50000,
  num_customers: 300,
};

const FIELDS = [
  { key: 'start_month_index', label: 'Start month index', hint: '25 = Jan of year 3', step: 1 },
  { key: 'months_ahead', label: 'Months to forecast', hint: '1 to 24', step: 1 },
  { key: 'last_known_sales', label: 'Last month sales (LKR)', hint: 'Starting point', step: 1000 },
  { key: 'marketing_spend', label: 'Marketing spend / month (LKR)', hint: 'Training range 20K to 80K', step: 1000 },
  { key: 'num_customers', label: 'Customers / month', hint: 'Training range 150 to 500', step: 10 },
];

/**
 * SalesTrends (route: "/sales")
 * Includes a forecast simulator: change the inputs, press "Run forecast",
 * and the chart, stat cards and table update from the Flask model
 * (/predict-series). If the backend is offline it shows mock data.
 */
export default function SalesTrends() {
  const [salesTrend, setSalesTrend] = useState(mockSalesTrend);
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rawForecast, setRawForecast] = useState([]);

  const runForecast = (values) => {
    setLoading(true);
    setError('');
    const payload = Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));
    predictSeries(payload)
      .then((res) => {
        setRawForecast(res.forecast);
        setSalesTrend(
          res.forecast.map((row) => ({
            month: row.month,
            revenue: row.predicted_sales,
            target: Math.round(row.predicted_sales * 1.1), // placeholder target: +10%
          }))
        );
        setLive(true);
      })
      .catch((err) => {
        console.error('Prediction request failed:', err);
        setLive(false);
        setError('Could not reach the backend. Is "python app.py" running on port 5000?');
      })
      .finally(() => setLoading(false));
  };

  // Run once on page load with the default inputs
  useEffect(() => {
    runForecast(DEFAULT_INPUTS);
  }, []);

  const handleChange = (key, value) => setInputs((prev) => ({ ...prev, [key]: value }));
  const handleReset = () => {
    setInputs(DEFAULT_INPUTS);
    runForecast(DEFAULT_INPUTS);
  };

  const totalRevenue = salesTrend.reduce((sum, m) => sum + m.revenue, 0);
  const totalTarget = salesTrend.reduce((sum, m) => sum + m.target, 0);
  const gapPct = Math.round(((totalTarget - totalRevenue) / totalTarget) * 100);
  const bestMonth = salesTrend.reduce((a, b) => (b.revenue > a.revenue ? b : a));
  const growth = Math.round(
    ((salesTrend[salesTrend.length - 1].revenue - salesTrend[0].revenue) / salesTrend[0].revenue) * 100
  );

  return (
    <>
      <PageHeader
        caseRef="Exhibit · Revenue"
        title="Sales trends"
        subtitle={live ? 'Live model forecast from the Flask backend' : 'Showing sample data (backend offline)'}
      />

      <SectionCard
        eyebrow="Model test"
        title="Forecast simulator"
        subtitle="Change the inputs and run the trained model"
        className="rise-in mb-4"
        action={
          <span className={`font-mono text-[10.5px] tracking-[0.1em] uppercase ${live ? 'text-vital' : 'text-pulse'}`}>
            {live ? '● Backend connected' : '● Backend offline'}
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
        </div>

        <div className="flex items-center gap-3 mt-4">
          <button
            type="button"
            onClick={() => runForecast(inputs)}
            disabled={loading}
            className="inline-flex items-center justify-center bg-vital text-white text-[13px] font-medium px-4 py-2 rounded-md hover:bg-vital-deep active:scale-[0.97] transition-all disabled:opacity-60"
          >
            {loading ? 'Running…' : 'Run forecast'}
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="text-[13px] text-ink-soft hover:text-ink px-3 py-2"
          >
            Reset
          </button>
          {error && <span className="text-[12px] text-pulse">{error}</span>}
        </div>
      </SectionCard>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <div className="rise-in"><StatCard label="Total revenue" value={`LKR ${(totalRevenue / 1000).toFixed(0)}K`} /></div>
        <div className="rise-in" style={{ '--delay': '60ms' }}><StatCard label="Gap to target" value={`${gapPct}%`} delta="Below plan" trend="down" /></div>
        <div className="rise-in" style={{ '--delay': '120ms' }}><StatCard label="Best month" value={bestMonth.month} delta={`LKR ${(bestMonth.revenue / 1000).toFixed(0)}K`} trend="up" /></div>
        <div className="rise-in" style={{ '--delay': '180ms' }}><StatCard label={`${salesTrend.length}-month growth`} value={`${growth}%`} delta={growth >= 0 ? 'Trending up' : 'Trending down'} trend={growth >= 0 ? 'up' : 'down'} /></div>
      </div>

      <SectionCard eyebrow="Exhibit A" title="Revenue vs target" subtitle="Monthly, LKR" className="rise-in mb-4" style={{ '--delay': '220ms' }}>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={salesTrend} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#233047" vertical={false} />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: '#8493AD' }} />
              <YAxis
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 11, fill: '#8493AD' }}
                tickFormatter={(v) => `${Math.round(v / 1000)}K`}
              />
              <Tooltip
                formatter={(v) => `LKR ${Number(v).toLocaleString()}`}
                contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #233047', background: '#131C2E', color: '#E9EEF9', fontFamily: 'IBM Plex Mono, monospace' }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="target" name="Target" stroke="#3A4A66" strokeDasharray="4 4" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="revenue" name="Revenue" stroke="#3E7BFA" strokeWidth={2.5} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      {rawForecast.length > 0 && (
        <SectionCard eyebrow="Model output" title="Predicted sales by month" subtitle="Exact values returned by /predict-series" className="rise-in mb-4">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left font-mono text-[10px] tracking-[0.1em] uppercase text-ink-faint border-b border-line">
                  <th className="py-2 pr-4">Month index</th>
                  <th className="py-2 pr-4">Month</th>
                  <th className="py-2 text-right">Predicted sales (LKR)</th>
                </tr>
              </thead>
              <tbody>
                {rawForecast.map((row) => (
                  <tr key={row.month_index} className="border-b border-line/50 text-ink">
                    <td className="py-2 pr-4 font-mono">{row.month_index}</td>
                    <td className="py-2 pr-4">{row.month}</td>
                    <td className="py-2 text-right font-mono">{Math.round(row.predicted_sales).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      <SectionCard eyebrow="Analyst note" title="Insight" className="rise-in" style={{ '--delay': '280ms' }}>
        <p className="text-[13px] text-ink-soft leading-relaxed">
          Revenue has stayed below target every month, with the gap widening in March and April. The afternoon
          promotion recommendation in the diagnostic report targets the 2pm-5pm slow window identified in this
          period, which is the largest single contributor to the shortfall.
        </p>
      </SectionCard>
    </>
  );
}
