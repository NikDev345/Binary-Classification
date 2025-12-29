# ================================
# CREDIT CARD FRAUD DETECTION
# Binary Classification Project
# ================================

# STEP 1: IMPORT LIBRARIES
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# STEP 2: LOAD DATA
df = pd.read_csv("creditcard.csv")
print("Dataset Loaded Successfully\n")


# STEP 3: CHECK DATA
print("First 5 rows:")
print(df.head(), "\n")

print("Class distribution:")
print(df['Class'].value_counts(), "\n")


# STEP 4: FEATURES & LABEL
X = df.drop("Class", axis=1)
y = df["Class"]


# STEP 5: TRAIN-TEST SPLIT (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Done\n")


# STEP 6: BASELINE MODEL (LOGISTIC REGRESSION)
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

print("=== BASELINE MODEL RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# STEP 7: CONFUSION MATRIX VISUALIZATION
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix - Baseline Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# STEP 8: HANDLE IMBALANCE USING SMOTE
smote = SMOTE(random_state=10)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_resampled).value_counts(), "\n")


# STEP 9: TRAIN MODEL AGAIN (AFTER SMOTE)
smote_model = LogisticRegression(max_iter=1000)
smote_model.fit(X_resampled, y_resampled)

y_pred_smote = smote_model.predict(X_test)

print("=== MODEL RESULTS AFTER SMOTE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_smote))
print("Classification Report:")
print(classification_report(y_test, y_pred_smote))


# STEP 10: CONFUSION MATRIX AFTER SMOTE
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt="d")
plt.title("Confusion Matrix - After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("Project Completed Successfully")
