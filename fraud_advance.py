import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import shap

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.2

# Business costs
COST_FALSE_NEGATIVE = 100   # Missed fraud
COST_FALSE_POSITIVE = 5     # Blocked legit transaction

LOW_RISK_THRESHOLD = 0.20
HIGH_RISK_THRESHOLD = 0.80

# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset...")

df = pd.read_csv("creditcard.csv")
df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)

# Time-aware split (IMPORTANT)
df = df.sort_values("Time")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Class distribution:\n", y.value_counts(), "\n")

# =========================================================
# TIME-BASED TRAIN–TEST SPLIT
# =========================================================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# =========================================================
# CLASS WEIGHT (COST SENSITIVE)
# =========================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# =========================================================
# PIPELINE: SMOTE + XGBOOST
# =========================================================
pipeline = Pipeline(steps=[
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# =========================================================
# TRAIN MODEL
# =========================================================
print("Training model...")
pipeline.fit(X_train, y_train)

# =========================================================
# PROBABILITIES
# =========================================================
y_probs = pipeline.predict_proba(X_test)[:, 1]

# =========================================================
# METRICS
# =========================================================
roc_auc = roc_auc_score(y_test, y_probs)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")

# =========================================================
# THRESHOLD OPTIMIZATION (F1)
# =========================================================
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal F1 Threshold: {best_threshold:.3f}")

# =========================================================
# FINAL PREDICTIONS
# =========================================================
y_pred = (y_probs >= best_threshold).astype(int)

print("\n=== FINAL RESULTS ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================================================
# BUSINESS COST EVALUATION
# =========================================================
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)

print("Business Cost Evaluation:")
print(f"False Negatives Cost: {fn * COST_FALSE_NEGATIVE}")
print(f"False Positives Cost: {fp * COST_FALSE_POSITIVE}")
print(f"TOTAL COST: ₹{total_cost}")

# =========================================================
# DUAL-THRESHOLD RISK SYSTEM
# =========================================================
def risk_bucket(prob):
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH_RISK"
    elif prob >= LOW_RISK_THRESHOLD:
        return "REVIEW"
    else:
        return "LOW_RISK"

risk_labels = [risk_bucket(p) for p in y_probs]
print("\nRisk Distribution:")
print(pd.Series(risk_labels).value_counts())

# =========================================================
# CONFUSION MATRIX VISUAL
# =========================================================
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix – XGBoost (Optimized)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# PRECISION–RECALL CURVE
# =========================================================
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================
print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(pipeline.named_steps["model"])
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")

# =========================================================
# REAL-TIME SCORING FUNCTION
# =========================================================
def score_transaction(transaction_df):
    """
    transaction_df: pandas DataFrame with single transaction
    """
    prob = pipeline.predict_proba(transaction_df)[0, 1]
    decision = "BLOCK" if prob >= best_threshold else "ALLOW"
    risk = risk_bucket(prob)
    return {
        "fraud_probability": float(prob),
        "decision": decision,
        "risk_level": risk
    }

print("\n✅ Advanced fraud detection system completed successfully.")
