"""
Fraud Detection ML Pipeline — Adapted for real dataset
Columns: TransactionID, TransactionDate, Amount, MerchantID,
         TransactionType, Location, IsFraud
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, recall_score, f1_score)
import joblib, json, warnings
warnings.filterwarnings("ignore")

print("=" * 58)
print("  FRAUD DETECTION PIPELINE  —  Real Dataset")
print("=" * 58)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading dataset...")

df = pd.read_csv("fraud_with_predictions.csv")
print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
print(f"  Fraud rate: {df['IsFraud'].mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 2: Clean + Preprocess
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Cleaning & Preprocessing...")

df.drop_duplicates(subset="TransactionID", inplace=True)
df.dropna(inplace=True)

# Parse date — format is DD-MM-YYYY
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True, errors="coerce")
df.dropna(subset=["TransactionDate"], inplace=True)

# Clip extreme amounts
upper = df["Amount"].quantile(0.999)
df["Amount"] = df["Amount"].clip(upper=upper)

print(f"  Clean dataset: {len(df):,} rows, {df.isnull().sum().sum()} nulls")

# ─────────────────────────────────────────────────────────────
# STEP 2b: Feature Engineering
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2b] Feature Engineering...")

df["day_of_week"] = df["TransactionDate"].dt.dayofweek
df["month"]       = df["TransactionDate"].dt.month
df["day"]         = df["TransactionDate"].dt.day
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["quarter"]     = df["TransactionDate"].dt.quarter

scaler = StandardScaler()
df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["amount_bucket"] = pd.cut(
    df["Amount"],
    bins=[0, 500, 1000, 2000, 3500, 9999],
    labels=[0, 1, 2, 3, 4]
).astype(int)

df["type_enc"] = (df["TransactionType"] == "refund").astype(int)

le_loc = LabelEncoder()
df["location_enc"] = le_loc.fit_transform(df["Location"])

merchant_freq = df["MerchantID"].value_counts().to_dict()
df["merchant_freq"] = df["MerchantID"].map(merchant_freq)

print("  Features created: day_of_week, month, day, is_weekend, quarter,")
print("                    amount_scaled, amount_bucket, type_enc,")
print("                    location_enc, merchant_freq")

df.to_csv("fraud_enriched.csv", index=False)

# ─────────────────────────────────────────────────────────────
# STEP 3: Model Training
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Training Models...")

FEATURES = [
    "Amount", "amount_scaled", "amount_bucket",
    "day_of_week", "month", "day", "is_weekend", "quarter",
    "MerchantID", "merchant_freq",
    "type_enc", "location_enc"
]
TARGET = "IsFraud"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
print("  Training Random Forest (200 trees)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=14, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_proba)

print(f"  Random Forest trained")
print(f"     Precision : {prec:.4f}")
print(f"     Recall    : {rec:.4f}")
print(f"     F1-Score  : {f1:.4f}")
print(f"     ROC-AUC   : {auc:.4f}")

# Isolation Forest
print("  Training Isolation Forest (anomaly detection)...")
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_test = (iso.predict(X_test) == -1).astype(int)
print(f"  Isolation Forest | Anomalies in test set: {iso_test.sum():,}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion Matrix:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Predictions on full dataset
# ─────────────────────────────────────────────────────────────
print("\n[STEP 4] Generating predictions on full dataset...")
df["prediction"] = rf.predict(df[FEATURES])
df["fraud_prob"] = rf.predict_proba(df[FEATURES])[:, 1]
df["iso_anomaly"] = (iso.predict(df[FEATURES]) == -1).astype(int)
df.to_csv("fraud_with_predictions.csv", index=False)
print(f"  Flagged: {df['prediction'].sum():,} / {len(df):,} transactions")

# ─────────────────────────────────────────────────────────────
# STEP 5: Save artifacts
# ─────────────────────────────────────────────────────────────
print("\n[STEP 5] Saving model artifacts...")
joblib.dump(rf,     "fraud_model.pkl")
joblib.dump(iso,    "iso_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_loc, "le_location.pkl")

metrics = {
    "precision": round(prec, 4), "recall": round(rec, 4),
    "f1": round(f1, 4), "roc_auc": round(auc, 4),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "total_train": int(len(X_train)), "total_test": int(len(X_test)),
    "features": FEATURES,
    "locations": le_loc.classes_.tolist(),
    "fraud_total": int(df["IsFraud"].sum()),
    "total_rows": int(len(df))
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("  fraud_model.pkl | iso_model.pkl | scaler.pkl | le_location.pkl")
print("  metrics.json | fraud_with_predictions.csv")
print("\n" + "=" * 58)
print("  PIPELINE COMPLETE")
print("=" * 58)
