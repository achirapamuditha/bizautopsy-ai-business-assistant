// Small fetch wrapper for the Flask backend (backend/app.py).
const API_BASE = 'http://127.0.0.1:5000';

async function post(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try {
      detail = (await res.json()).error || '';
    } catch (e) {
      /* response had no JSON body */
    }
    throw new Error(detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

// Model 1 - sales trend (Sales trends page)
export function predictSales(payload) {
  return post('/predict-sales', payload);
}
export function predictSeries(payload) {
  return post('/predict-series', payload);
}

// Model 2 - customer churn risk (Customers page)
export function predictChurn(payload) {
  return post('/predict-churn', payload);
}

// Model 3 - business health score (Diagnostics page)
export function predictHealth(payload) {
  return post('/predict-health', payload);
}
