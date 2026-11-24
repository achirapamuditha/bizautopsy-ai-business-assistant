"""
BizAutopsy AI backend (Flask) - serves all three trained models.

Train each model once, from this folder:
    python sales_trend_model.py
    python churn_model.py
    python health_model.py
Then start the server:
    python app.py        (http://127.0.0.1:5000)

Endpoints
  GET  /health           -> status + which models are loaded
  POST /predict-sales    -> one month's sales prediction
  POST /predict-series   -> multi-month sales forecast   (Sales trends page)
  POST /predict-churn    -> churn probability for a customer (Customers page)
  POST /predict-health   -> business health score 0-100     (Diagnostics page)

Every route must be defined ABOVE the `if __name__ == "__main__":` block.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

HERE = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])


# ---------------------------------------------------------------------------
# Model loading. A missing model does not crash the server: its endpoint just
# answers 503 telling you which script to run.
# ---------------------------------------------------------------------------
def load_bundle(prefix):
    try:
        return {
            "model": joblib.load(HERE / f"{prefix}_model.pkl"),
            "scaler": joblib.load(HERE / f"{prefix}_scaler.pkl"),
            "meta": json.loads((HERE / f"{prefix}_meta.json").read_text()),
        }
    except FileNotFoundError:
        return None


SALES = load_bundle("sales_trend")
CHURN = load_bundle("churn")
HEALTH = load_bundle("health")

TRAIN_SCRIPTS = {"sales_trend": "sales_trend_model.py", "churn": "churn_model.py", "health": "health_model.py"}


def not_trained(prefix):
    return jsonify({"error": f"Model not trained yet. Run: python {TRAIN_SCRIPTS[prefix]}, then restart app.py"}), 503


def to_input(bundle, values):
    """Build the model input from a dict of feature values (in training order)."""
    row = pd.DataFrame([values])[bundle["meta"]["features"]]
    return bundle["scaler"].transform(row) if bundle["meta"]["needs_scaling"] else row


def read_features(bundle, data):
    """Pull every required feature out of the request JSON as floats."""
    return {f: float(data[f]) for f in bundle["meta"]["features"]}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "sales_trend": SALES["meta"]["model"] if SALES else None,
            "churn": CHURN["meta"]["model"] if CHURN else None,
            "health": HEALTH["meta"]["model"] if HEALTH else None,
        },
    })


# ---------------------------------------------------------------------------
# Model 1: sales trend
# ---------------------------------------------------------------------------
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def is_festive(month_index):
    return 1 if month_index % 12 in (4, 11, 0) else 0  # same rule as training


def predict_one_sale(month_index, prev_sales, marketing, customers, festive):
    values = {
        "month_index": month_index,
        "prev_month_sales": prev_sales,
        "marketing_spend": marketing,
        "num_customers": customers,
        "is_festive_season": festive,
    }
    return float(SALES["model"].predict(to_input(SALES, values))[0])


@app.route("/predict-sales", methods=["POST"])
def predict_sales():
    if SALES is None:
        return not_trained("sales_trend")
    data = request.get_json(silent=True) or {}
    try:
        month_index = int(data["month_index"])
        pred = predict_one_sale(
            month_index,
            float(data["prev_month_sales"]),
            float(data["marketing_spend"]),
            float(data["num_customers"]),
            int(data.get("is_festive_season", is_festive(month_index))),
        )
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400
    return jsonify({"prediction": pred})


@app.route("/predict-series", methods=["POST"])
def predict_series():
    if SALES is None:
        return not_trained("sales_trend")
    data = request.get_json(silent=True) or {}
    try:
        months_ahead = int(data.get("months_ahead", 6))
        start = int(data["start_month_index"])
        prev_sales = float(data["last_known_sales"])
        marketing = float(data["marketing_spend"])
        customers = float(data["num_customers"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    if not 1 <= months_ahead <= 24:
        return jsonify({"error": "months_ahead must be between 1 and 24"}), 400

    forecast = []
    for i in range(months_ahead):
        idx = start + i
        pred = predict_one_sale(idx, prev_sales, marketing, customers, is_festive(idx))
        forecast.append({
            "month_index": idx,
            "month": MONTH_NAMES[(idx - 1) % 12],
            "predicted_sales": round(pred, 2),
        })
        prev_sales = pred
    return jsonify({"forecast": forecast})


# ---------------------------------------------------------------------------
# Model 2: customer churn
# ---------------------------------------------------------------------------
@app.route("/predict-churn", methods=["POST"])
def predict_churn():
    if CHURN is None:
        return not_trained("churn")
    data = request.get_json(silent=True) or {}
    try:
        values = read_features(CHURN, data)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    prob = float(CHURN["model"].predict_proba(to_input(CHURN, values))[0, 1])
    level = "Low" if prob < 0.35 else "Medium" if prob < 0.65 else "High"
    return jsonify({
        "churn_probability": round(prob, 4),
        "risk_level": level,
        "model": CHURN["meta"]["model"],
    })


# ---------------------------------------------------------------------------
# Model 3: business health score
# ---------------------------------------------------------------------------
@app.route("/predict-health", methods=["POST"])
def predict_health():
    if HEALTH is None:
        return not_trained("health")
    data = request.get_json(silent=True) or {}
    try:
        values = read_features(HEALTH, data)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    score = float(np.clip(HEALTH["model"].predict(to_input(HEALTH, values))[0], 0, 100))
    level = "High risk" if score < 40 else "Moderate risk" if score < 70 else "Low risk"
    return jsonify({
        "health_score": round(score, 1),
        "risk_level": level,
        "model": HEALTH["meta"]["model"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
