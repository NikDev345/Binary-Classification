import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE = 42
SAMPLE_FRACTION = 0.2

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("creditcard.csv")
df = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)

X = df.drop("Class", axis=1)
y = df["Class"]

print("Class distribution:\n", y.value_counts(), "\n")

# =========================================================
# TRAIN–TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# =========================================================
# COMPUTE CLASS WEIGHT (IMPORTANT)
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
pipeline.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
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

print(f"Optimal threshold: {best_threshold:.3f}")

# =========================================================
# FINAL PREDICTIONS
# =========================================================
y_pred = (y_probs >= best_threshold).astype(int)

print("\n=== FINAL RESULTS ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix – XGBoost")
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
plt.title("Precision–Recall Curve (XGBoost)")
plt.show()

print("\n✅ Advanced XGBoost fraud model completed successfully.")
