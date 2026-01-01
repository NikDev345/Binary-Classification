import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)

from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

# Sampling for performance (engineering decision)
SAMPLE_FRACTION = 0.2   # change to 1.0 for full dataset
df = df.sample(frac=SAMPLE_FRACTION, random_state=42)

print("Dataset loaded successfully")
print("Class distribution:")
print(df["Class"].value_counts(), "\n")

# ---------------------------------------------------------
# STEP 2: FEATURES & TARGET
# ---------------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------------------------------------------------
# STEP 3: STRATIFIED TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# PART 1: BASELINE MODEL (BEFORE SMOTE)
# =========================================================
print("Training baseline model (Before SMOTE)...")

baseline_model = LogisticRegression(
    max_iter=300,
    solver="liblinear",
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

y_pred_before = baseline_model.predict(X_test)
y_probs_before = baseline_model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS BEFORE SMOTE ===")
print(confusion_matrix(y_test, y_pred_before))
print(classification_report(y_test, y_pred_before))

# Confusion Matrix - Before SMOTE
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_before), annot=True, fmt="d")
plt.title("Confusion Matrix - Before SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# PART 2: APPLY SMOTE
# =========================================================
print("Applying SMOTE...")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts(), "\n")

# =========================================================
# PART 3: MODEL AFTER SMOTE
# =========================================================
print("Training model after SMOTE...")

model = LogisticRegression(
    max_iter=300,
    solver="liblinear",
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

y_pred_after = model.predict(X_test)
y_probs_after = model.predict_proba(X_test)[:, 1]

print("\n=== RESULTS AFTER SMOTE ===")
print(confusion_matrix(y_test, y_pred_after))
print(classification_report(y_test, y_pred_after))

# Confusion Matrix - After SMOTE
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_after), annot=True, fmt="d")
plt.title("Confusion Matrix - After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# PART 4: ROC–AUC (AFTER SMOTE)
# =========================================================
roc_auc = roc_auc_score(y_test, y_probs_after)
print("ROC–AUC Score (After SMOTE):", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_probs_after)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - After SMOTE")
plt.legend()
plt.show()

# =========================================================
# PART 5: PRECISION–RECALL AUC
# =========================================================
precision, recall, _ = precision_recall_curve(y_test, y_probs_after)
pr_auc = auc(recall, precision)

print("PR–AUC Score (After SMOTE):", pr_auc)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve - After SMOTE")
plt.show()

# =========================================================
# PART 6: PROBABILITY CALIBRATION
# =========================================================
print("Calibrating probabilities...")

calibrated_model = CalibratedClassifierCV(
    estimator=model,  
    method="sigmoid"
)

calibrated_model.fit(X_train_smote, y_train_smote)

calib_probs = calibrated_model.predict_proba(X_test)[:, 1]

# =========================================================
# PART 7: BUSINESS-DRIVEN THRESHOLD
# =========================================================
BUSINESS_THRESHOLD = 0.25

y_business = (calib_probs >= BUSINESS_THRESHOLD).astype(int)

print("\n=== BUSINESS THRESHOLD RESULTS ===")
print("Threshold:", BUSINESS_THRESHOLD)
print(confusion_matrix(y_test, y_business))
print(classification_report(y_test, y_business))

# Final Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_business), annot=True, fmt="d")
plt.title("Confusion Matrix - Business Threshold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n✅ Final fraud detection pipeline completed successfully.")
