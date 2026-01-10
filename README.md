🚨 Credit Card Fraud Detection
Binary Classification using Machine Learning
📌 Project Overview

Credit card fraud detection is a critical real-world problem where fraudulent transactions are extremely rare compared to legitimate ones.
This project builds an end-to-end binary classification system to identify fraudulent credit card transactions using machine learning, while carefully handling:

Severe class imbalance

Performance constraints

Business-driven evaluation metrics

This is not just model training — it demonstrates engineering-level ML thinking.

🎯 Problem Statement

Given a dataset of credit card transactions, classify each transaction as:

0 → Legitimate Transaction

1 → Fraudulent Transaction

This is a binary classification problem with highly imbalanced data, making it realistic and challenging.

📊 Dataset Information

Dataset Name: Credit Card Fraud Detection

Source: Kaggle

File Name: creditcard.csv

Total Transactions: ~284,000

Fraudulent Transactions: ~492

📥 Dataset Download Instructions

Go to Kaggle

Search for “Credit Card Fraud Detection”

Download the dataset

Place the file in the project root as:

creditcard.csv

🧾 Dataset Description

Each row represents one credit card transaction

Features V1 to V28 are anonymized (PCA-transformed)

Amount represents the transaction value

Class is the target variable:

0 → Legitimate

1 → Fraud

⚠️ Key Challenge: Class Imbalance

Fraud transactions represent less than 0.2% of the data

Accuracy alone is misleading

Special techniques are required to detect fraud effectively

This project explicitly addresses this issue.

🧠 Machine Learning Approach
✔ Learning Type

Supervised Learning

✔ Problem Type

Binary Classification

✔ Models Used

Logistic Regression (Baseline)

Logistic Regression with SMOTE

XGBoost (Advanced Model)

⚙️ Project Workflow

Load and analyze the dataset

Apply sampling for hardware efficiency

Perform stratified train–test split

Train baseline model (Before SMOTE)

Apply SMOTE to balance the dataset

Train model after SMOTE

Evaluate using proper metrics

Calibrate probabilities

Apply business-driven decision threshold

Train and evaluate XGBoost advanced model

📈 Evaluation Metrics

To properly evaluate an imbalanced dataset, the following metrics are used:

Precision

Recall

F1-Score

Confusion Matrix

ROC–AUC

Precision–Recall AUC

📌 Recall is prioritized, as missing a fraudulent transaction is more costly than flagging a legitimate one.

🔍 Baseline Results (Before SMOTE)

The baseline Logistic Regression model is trained on the original imbalanced dataset.

📷 Confusion Matrix — Before SMOTE

🔹 Observation

Very high accuracy

Very poor fraud recall

Model heavily biased toward legitimate transactions

🔥 Improved Results (After SMOTE)

SMOTE (Synthetic Minority Over-sampling Technique) is applied before training.

📷 Confusion Matrix — After SMOTE

🔹 Observation

Improved fraud detection

Better recall for fraudulent transactions

More balanced learning

📊 ROC Curve & Precision–Recall Curve

The project includes:

ROC Curve → Measures class separation

Precision–Recall Curve → More informative for imbalanced data




🧠 Probability Calibration & Business Threshold

Model probabilities are calibrated to reflect realistic fraud risk

A custom decision threshold is applied instead of the default 0.5

Improves fraud detection based on business risk considerations

🚀 Advanced Model: XGBoost

To further improve fraud detection performance, an advanced gradient boosting model is added.

📌 Implementation Details

New file added:

xg_boost.py


Uses XGBoost for non-linear learning

Handles class imbalance using scale_pos_weight

Optimized using PR-AUC, not accuracy

📷 XGBoost Results
🔹 XGBoost Performance Visualization

🔹 Precision–Recall Curve (XGBoost)

🔹 Observation

Strong improvement in fraud detection

Better precision–recall tradeoff

Industry-grade performance for imbalanced classification

### 📷 Confusion Matrix — Before SMOTE

![Before SMOTE Confusion Matrix](images/before_smote.png)

🔥 Improved Results (After SMOTE)
### 📷 Confusion Matrix — After SMOTE

![After SMOTE Confusion Matrix](images/after_smote.png)

📊 ROC Curve & Precision–Recall Curve
### 📈 ROC Curve
![ROC Curve](images/roc_curve.png)

### 📉 Precision–Recall Curve
![PR Curve](images/pr_curve.png)

🧠 Probability Calibration & Business Threshold
### 🧠 Business-Driven Threshold

![Business Threshold](images/bussiness_threesold.png)

🚀 Advanced Model: XGBoost
## 🚀 Advanced Model: XGBoost

A new advanced model is implemented in `xg_boost.py` using XGBoost to capture non-linear fraud patterns.

📷 XGBoost Visualizations
### 📊 XGBoost Performance

![XGBoost Results](images/xg_boost.png)

### 📉 Precision–Recall Curve (XGBoost)

![PR XGBoost](images/pr_xgboost.png)
🖼️ Image Assets Summary (Optional but Professional)

## 🖼️ Visualization Assets

| Image | Description |
|------|-------------|
| `before_smote.png` | Confusion matrix before SMOTE |
| `after_smote.png` | Confusion matrix after SMOTE |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision–Recall curve |
| `bussiness_threesold.png` | Business threshold confusion matrix |
| `xg_boost.png` | XGBoost model performance |
| `pr_xgboost.png` | Precision–Recall curve for XGBoost |


🧰 Technologies & Libraries Used

Python

NumPy

Pandas

Scikit-learn

Imbalanced-learn (SMOTE)

XGBoost

Matplotlib

Seaborn

🚀 Key Learnings

Accuracy is unreliable for imbalanced datasets

Handling class imbalance is essential in fraud detection

SMOTE significantly improves minority class detection

Threshold tuning is more important than raw scores

Advanced models like XGBoost capture complex fraud patterns

Real-world ML requires balancing performance and constraints

🔮 Future Improvements

Ensemble learning (Logistic + XGBoost)

Hyperparameter tuning

Time-aware validation

SHAP explainability

Real-time fraud detection API

Web deployment using Streamlit

✅ Conclusion

This project demonstrates a complete, real-world fraud detection pipeline using binary classification.
It emphasizes correct evaluation, imbalance handling, and engineering decisions, making it suitable for:

Academic submission

GitHub portfolios

Technical interviews
