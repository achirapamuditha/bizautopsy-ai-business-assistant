"""
Model 2: Customer churn risk classifier (feeds the Customers page).

Run once from the backend folder:   python churn_model.py

Predicts the probability that a customer will NOT come back in the next 90 days.
Uses synthetic data (like the sales model) so it can be trained and demoed
without a real dataset. Swap the DATA section for a real CSV later.

Outputs (same folder as this script):
  churn_model.pkl, churn_scaler.pkl, churn_meta.json,
  churn_data_sample.csv, churn_model_results.png
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ DATA
n = 2000
days_since_last_visit = rng.integers(0, 121, n)                 # days
visits_last_90d = np.clip(rng.poisson(4, n), 0, 20)             # count
avg_basket_value = rng.uniform(500, 5000, n).round(0)           # LKR
loyalty_member = (rng.random(n) < 0.20).astype(int)             # 0/1
tenure_months = rng.integers(1, 49, n)                          # months

logit = (
    -1.0
    + 0.04 * (days_since_last_visit - 30)
    - 0.30 * (visits_last_90d - 4)
    - 0.0002 * (avg_basket_value - 1800)
    - 1.2 * loyalty_member
    - 0.02 * (tenure_months - 12)
    + rng.normal(0, 0.5, n)
)
prob = 1 / (1 + np.exp(-logit))
churned = (rng.random(n) < prob).astype(int)

FEATURES = ["days_since_last_visit", "visits_last_90d", "avg_basket_value",
            "loyalty_member", "tenure_months"]
df = pd.DataFrame({
    "days_since_last_visit": days_since_last_visit,
    "visits_last_90d": visits_last_90d,
    "avg_basket_value": avg_basket_value,
    "loyalty_member": loyalty_member,
    "tenure_months": tenure_months,
    "churned": churned,
})
df.to_csv(HERE / "churn_data_sample.csv", index=False)
print(f"Dataset saved -> churn_data_sample.csv  ({n} customers, churn rate {churned.mean():.1%})\n")

# ------------------------------------------------------------ SPLIT / SCALE
X, y = df[FEATURES], df["churned"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------------------------ TRAIN
log_model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X_train, y_train)


def report(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred), "\n")
    return {"accuracy": acc, "f1": f1, "auc": auc}


print("=" * 40)
log_prob = log_model.predict_proba(X_test_s)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]
log_scores = report("Logistic Regression", y_test, (log_prob >= 0.5).astype(int), log_prob)
rf_scores = report("Random Forest", y_test, (rf_prob >= 0.5).astype(int), rf_prob)

imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importance (Random Forest):")
print(imp, "\n")

# ------------------------------------------------------------------- PLOT
plt.figure(figsize=(6, 4))
imp.sort_values().plot(kind="barh")
plt.title("Churn model - feature importance (Random Forest)")
plt.tight_layout()
plt.savefig(HERE / "churn_model_results.png", dpi=150)

# ------------------------------------------------------------------- SAVE
if rf_scores["auc"] > log_scores["auc"]:
    best_model, best_name, needs_scaling, best = rf_model, "random_forest", False, rf_scores
else:
    best_model, best_name, needs_scaling, best = log_model, "logistic_regression", True, log_scores

joblib.dump(best_model, HERE / "churn_model.pkl")
joblib.dump(scaler, HERE / "churn_scaler.pkl")
(HERE / "churn_meta.json").write_text(json.dumps({
    "model": best_name,
    "needs_scaling": needs_scaling,
    "features": FEATURES,
    "metrics": {k: round(float(v), 3) for k, v in best.items()},
}, indent=2))
print(f"Best model ({best_name}, AUC {best['auc']:.3f}) saved for the web app backend.")

# ---------------------------------------------------------------- EXAMPLE
example = pd.DataFrame([{
    "days_since_last_visit": 75, "visits_last_90d": 1, "avg_basket_value": 1500,
    "loyalty_member": 0, "tenure_months": 6,
}])[FEATURES]
inp = scaler.transform(example) if needs_scaling else example
print(f"Example (inactive 75 days, 1 visit, no loyalty): churn probability = {best_model.predict_proba(inp)[0, 1]:.1%}")
