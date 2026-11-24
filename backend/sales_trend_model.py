"""
Trains the sales trend models and saves them next to this file.

Run once from the backend folder:   python sales_trend_model.py

Outputs (all in the same folder as this script):
  sales_trend_model.pkl   -> the best model (by R2)
  sales_trend_scaler.pkl  -> the StandardScaler (always saved)
  sales_trend_meta.json   -> which model won + whether it needs scaling
  sales_data_sample.csv, sales_trend_prediction.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

# GENERATE / LOAD DATA
np.random.seed(42)
n_months = 24  # 2 years of monthly data

month_index = np.arange(1, n_months + 1)
is_festive_season = np.array([1 if m % 12 in [4, 11, 0] else 0 for m in month_index])
marketing_spend = np.random.uniform(20000, 80000, n_months)
num_customers = np.random.randint(150, 500, n_months)

base_trend = 300000 + month_index * 4000
seasonal_boost = is_festive_season * 60000
marketing_effect = marketing_spend * 1.8
customer_effect = num_customers * 350
noise = np.random.normal(0, 25000, n_months)

sales = base_trend + seasonal_boost + marketing_effect + customer_effect + noise
prev_month_sales = np.roll(sales, 1)
prev_month_sales[0] = sales[0] * 0.95

df = pd.DataFrame({
    "month_index": month_index,
    "prev_month_sales": prev_month_sales,
    "marketing_spend": marketing_spend,
    "num_customers": num_customers,
    "is_festive_season": is_festive_season,
    "sales": sales,
})
df.to_csv(HERE / "sales_data_sample.csv", index=False)
print("Sample dataset saved -> sales_data_sample.csv")

# TRAIN / TEST SPLIT
FEATURES = ["month_index", "prev_month_sales", "marketing_spend", "num_customers", "is_festive_season"]
X = df[FEATURES]
y = df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN MODELS
lr_model = LinearRegression().fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

gbr_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)
gbr_pred = gbr_model.predict(X_test)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE : LKR {mae:,.0f}")
    print(f"RMSE: LKR {rmse:,.0f}")
    print(f"R2  : {r2:.3f}\n")
    return mae, rmse, r2


print("=" * 40)
lr_scores = report("Linear Regression", y_test, lr_pred)
gbr_scores = report("Gradient Boosting Regressor", y_test, gbr_pred)

importance = pd.Series(gbr_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importance (Gradient Boosting):")
print(importance, "\n")

# PLOT
plt.figure(figsize=(8, 5))
plt.plot(y_test.values, marker="o", label="Actual Sales")
plt.plot(lr_pred, marker="x", linestyle="--", label="Linear Regression Prediction")
plt.plot(gbr_pred, marker="s", linestyle="--", label="Gradient Boosting Prediction")
plt.title("Sales Trend Model - Actual vs Predicted")
plt.xlabel("Test sample index")
plt.ylabel("Sales (LKR)")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "sales_trend_prediction.png", dpi=150)
print("Plot saved -> sales_trend_prediction.png")

# SAVE BEST MODEL (by R2) + scaler ALWAYS + meta so app.py knows what to do
if gbr_scores[2] > lr_scores[2]:
    best_model, best_name, needs_scaling = gbr_model, "gradient_boosting", False
else:
    best_model, best_name, needs_scaling = lr_model, "linear_regression", True

joblib.dump(best_model, HERE / "sales_trend_model.pkl")
joblib.dump(scaler, HERE / "sales_trend_scaler.pkl")
(HERE / "sales_trend_meta.json").write_text(
    json.dumps({"model": best_name, "needs_scaling": needs_scaling, "features": FEATURES}, indent=2)
)
print(f"\nBest model ({best_name}) saved for the web app backend.")

# EXAMPLE
next_month = pd.DataFrame({
    "month_index": [n_months + 1],
    "prev_month_sales": [sales[-1]],
    "marketing_spend": [50000],
    "num_customers": [300],
    "is_festive_season": [0],
})[FEATURES]
inp = scaler.transform(next_month) if needs_scaling else next_month
print(f"Predicted sales for month {n_months + 1}: LKR {best_model.predict(inp)[0]:,.0f}")
