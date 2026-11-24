"""
Model 3: Business health score regressor (feeds the Diagnostics page).

Run once from the backend folder:   python health_model.py

Predicts an overall business health score (0-100) from key business ratios.
Uses synthetic data so it can be trained and demoed without a real dataset.
Score bands match HealthGauge.jsx:  >= 70 Low risk, 40-69 Moderate, < 40 High.

Outputs (same folder as this script):
  health_model.pkl, health_scaler.pkl, health_meta.json,
  health_data_sample.csv, health_model_results.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ DATA
n = 1500
revenue_growth_pct = rng.uniform(-20, 25, n)          # % change vs last period
expense_ratio_pct = rng.uniform(55, 115, n)           # expenses as % of revenue
repeat_customer_rate_pct = rng.uniform(20, 80, n)     # % of customers who return
rent_share_pct = rng.uniform(5, 40, n)                # rent & utilities as % of expenses
inventory_days = rng.uniform(10, 120, n)              # days of stock on hand
loyalty_active = (rng.random(n) < 0.30).astype(int)   # 0/1

score = (
    50
    + 0.8 * revenue_growth_pct
    - 0.7 * (expense_ratio_pct - 85)
    + 0.35 * (repeat_customer_rate_pct - 50)
    - 0.5 * (rent_share_pct - 15)
    - 0.12 * (inventory_days - 45)
    + 4 * loyalty_active
    + rng.normal(0, 4, n)
)
health_score = np.clip(score, 0, 100)

FEATURES = ["revenue_growth_pct", "expense_ratio_pct", "repeat_customer_rate_pct",
            "rent_share_pct", "inventory_days", "loyalty_active"]
df = pd.DataFrame({
    "revenue_growth_pct": revenue_growth_pct,
    "expense_ratio_pct": expense_ratio_pct,
    "repeat_customer_rate_pct": repeat_customer_rate_pct,
    "rent_share_pct": rent_share_pct,
    "inventory_days": inventory_days,
    "loyalty_active": loyalty_active,
    "health_score": health_score,
}).round(2)
df.to_csv(HERE / "health_data_sample.csv", index=False)
print(f"Dataset saved -> health_data_sample.csv  ({n} businesses)\n")

# ------------------------------------------------------------ SPLIT / SCALE
X, y = df[FEATURES], df["health_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------ TRAIN
lr_model = LinearRegression().fit(X_train_s, y_train)
gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test_s)
gbr_pred = gbr_model.predict(X_test)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE : {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R2  : {r2:.3f}\n")
    return {"mae": mae, "rmse": rmse, "r2": r2}


print("=" * 40)
lr_scores = report("Linear Regression", y_test, lr_pred)
gbr_scores = report("Gradient Boosting Regressor", y_test, gbr_pred)

imp = pd.Series(gbr_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importance (Gradient Boosting):")
print(imp, "\n")

# ------------------------------------------------------------------- PLOT
plt.figure(figsize=(6, 5))
plt.scatter(y_test, lr_pred, s=10, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, gbr_pred, s=10, alpha=0.6, label="Gradient Boosting")
plt.plot([0, 100], [0, 100], "k--", linewidth=1)
plt.xlabel("Actual health score")
plt.ylabel("Predicted health score")
plt.title("Health score model - actual vs predicted")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "health_model_results.png", dpi=150)

# ------------------------------------------------------------------- SAVE
if gbr_scores["r2"] > lr_scores["r2"]:
    best_model, best_name, needs_scaling, best = gbr_model, "gradient_boosting", False, gbr_scores
else:
    best_model, best_name, needs_scaling, best = lr_model, "linear_regression", True, lr_scores

joblib.dump(best_model, HERE / "health_model.pkl")
joblib.dump(scaler, HERE / "health_scaler.pkl")
(HERE / "health_meta.json").write_text(json.dumps({
    "model": best_name,
    "needs_scaling": needs_scaling,
    "features": FEATURES,
    "metrics": {k: round(float(v), 3) for k, v in best.items()},
}, indent=2))
print(f"Best model ({best_name}, R2 {best['r2']:.3f}) saved for the web app backend.")

# ---------------------------------------------------------------- EXAMPLE
example = pd.DataFrame([{
    "revenue_growth_pct": 3.1, "expense_ratio_pct": 78, "repeat_customer_rate_pct": 51,
    "rent_share_pct": 28, "inventory_days": 70, "loyalty_active": 0,
}])[FEATURES]
inp = scaler.transform(example) if needs_scaling else example
print(f"Example (Silva Grocery-like profile): health score = {np.clip(best_model.predict(inp)[0], 0, 100):.1f}")
