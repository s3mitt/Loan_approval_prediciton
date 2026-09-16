"""
Loan Approval Prediction — v2 (uploaded dataset: loan_approval_dataset.csv)
=============================================================================
End-to-end ML pipeline: load -> clean/preprocess -> train -> evaluate -> save artifacts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

RANDOM_STATE = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("loan_approval_dataset.csv")

print("=" * 70)
print("1. RAW DATA OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")

# --- Clean up messy column names / string values (dataset has leading spaces) ---
df.columns = df.columns.str.strip()
str_cols = df.select_dtypes(include="object").columns
for c in str_cols:
    df[c] = df[c].str.strip()

print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTarget distribution:")
print(df["loan_status"].value_counts())

# ---------------------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. CLEANING & PREPROCESSING")
print("=" * 70)

df = df.drop(columns=["loan_id"])  # unique identifier, not predictive

# No missing values in this dataset, but keep a safety-net impute in case of
# future/unseen data with nulls (median for numeric, mode for categorical).
num_cols_all = df.select_dtypes(include=np.number).columns
for c in num_cols_all:
    df[c] = df[c].fillna(df[c].median())
cat_cols_all = ["education", "self_employed"]
for c in cat_cols_all:
    df[c] = df[c].fillna(df[c].mode()[0])

# --- Feature engineering ---
df["total_assets_value"] = (
    df["residential_assets_value"] + df["commercial_assets_value"]
    + df["luxury_assets_value"] + df["bank_asset_value"]
)
# Loan amount relative to annual income -> affordability signal
df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"].replace(0, np.nan)
df["loan_to_income_ratio"] = df["loan_to_income_ratio"].fillna(df["loan_to_income_ratio"].median())
# Loan amount relative to total assets -> collateral/coverage signal
df["loan_to_assets_ratio"] = df["loan_amount"] / df["total_assets_value"].replace(0, np.nan)
df["loan_to_assets_ratio"] = df["loan_to_assets_ratio"].fillna(df["loan_to_assets_ratio"].median())
# Log-transform large skewed monetary columns
for c in ["income_annum", "loan_amount", "total_assets_value"]:
    df[f"{c}_log"] = np.log1p(df[c])

# --- Encode categorical / target variables ---
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

print("Any missing after cleaning/encoding:", df.isnull().sum().sum())
print("\nFinal columns:", list(df.columns))

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
drop_for_features = ["loan_status", "residential_assets_value", "commercial_assets_value",
                      "luxury_assets_value", "bank_asset_value",  # kept as total_assets_value
                      "income_annum", "loan_amount", "total_assets_value"]  # keep *_log versions
feature_cols = [c for c in df.columns if c not in drop_for_features]

X = df[feature_cols]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

numeric_cols = ["no_of_dependents", "loan_term", "cibil_score", "loan_to_income_ratio",
                 "loan_to_assets_ratio", "income_annum_log", "loan_amount_log",
                 "total_assets_value_log"]
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. MODEL TRAINING")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                             random_state=RANDOM_STATE),
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="accuracy")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
    }
    results[name] = {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print(results[name]["report"])

# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE / COEFFICIENTS
# ---------------------------------------------------------------------------
rf_importances = pd.Series(
    results["Random Forest"]["model"].feature_importances_, index=X.columns
).sort_values(ascending=False)

lr_coefs = pd.Series(
    results["Logistic Regression"]["model"].coef_[0], index=X.columns
).sort_values(key=abs, ascending=False)

print("\nRandom Forest feature importances:")
print(rf_importances)
print("\nLogistic Regression coefficients (sorted by magnitude):")
print(lr_coefs)

# ---------------------------------------------------------------------------
# 6. SAVE PLOTS
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
best = results[best_name]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.heatmap(best["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"],
            ax=axes[0, 0])
axes[0, 0].set_title(f"Confusion Matrix - {best_name}")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].legend()

rf_importances.plot(kind="barh", ax=axes[1, 0], color="teal")
axes[1, 0].invert_yaxis()
axes[1, 0].set_title("Random Forest Feature Importance")

metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
comp_df = pd.DataFrame({name: [res["metrics"][m] for m in metric_names]
                         for name, res in results.items()}, index=metric_names)
comp_df.plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_title("Model Comparison")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend(loc="lower right")
axes[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=130)
print("\nSaved plots to model_evaluation.png")