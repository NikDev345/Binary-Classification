🚨 CREDIT CARD FRAUD DETECTION
Binary Classification using Machine Learning
📌 Project Overview

Credit card fraud detection is a critical real-world problem where fraudulent transactions are extremely rare compared to legitimate ones.
This project builds an end-to-end binary classification system to identify fraudulent credit card transactions using machine learning, while carefully handling class imbalance, performance constraints, and real-world evaluation metrics.

This is a Machine Learning Model 
using python lib 
that help us to find the Binary classifier 
it will Help Dataset to refine and clear the cluster 

Machine Learning Model Using Python Lib to classify dataset

The project demonstrates engineering-level ML thinking, not just model training.

🎯 Problem Statement

Given a dataset of credit card transactions, classify each transaction as:

0 → Legitimate Transaction

1 → Fraudulent Transaction

This is a binary classification problem with severe class imbalance, making it challenging and realistic.

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

Place the file in the project as:

creditcard.csv

🧾 Dataset Description

Each row represents one credit card transaction

Features V1 to V28 are anonymized (PCA-transformed)

Amount represents the transaction value

Class is the target variable:

0 → Legit

1 → Fraud

⚠️ Key Challenge: Class Imbalance

Fraud transactions represent less than 0.2% of the data

Accuracy alone is misleading

Special techniques are required to correctly detect fraud

This project explicitly addresses this issue.

🧠 Machine Learning Approach
✔ Learning Type

Supervised Learning

✔ Problem Type

Binary Classification

✔ Models Used

Logistic Regression (Baseline)

Logistic Regression with SMOTE

⚙️ Project Workflow

Load and analyze the dataset

Apply sampling to handle hardware limitations

Perform stratified train–test split

Train a baseline model (Before SMOTE)

Apply SMOTE to balance the dataset

Train model After SMOTE

Evaluate using appropriate metrics

Calibrate probabilities

Apply business-driven decision threshold

📈 Evaluation Metrics

To properly evaluate performance on an imbalanced dataset, the following metrics are used:

Precision

Recall

F1-Score

Confusion Matrix

ROC–AUC

Precision–Recall AUC

📌 Recall is prioritized, as missing a fraudulent transaction is more costly than flagging a legitimate one.

🔍 Baseline Results (Before SMOTE)

The baseline model is trained on the original imbalanced dataset to establish a reference.

📷 Confusion Matrix — Before SMOTE

![Before SMOTE Confusion Matrix](images/before_smote.png)


🔹 Observation:

Very high accuracy

Very poor fraud recall

Model biased toward legitimate transactions

🔥 Improved Results (After SMOTE)

SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset before training.

📷 Confusion Matrix — After SMOTE

![After SMOTE Confusion Matrix](images/after_smote.png)



🔹 Observation:

Improved fraud detection

Better recall for fraudulent transactions

More balanced learning

📊 ROC Curve & Precision–Recall Curve

The project includes:

ROC Curve to measure class separation

Precision–Recall Curve to evaluate performance on imbalanced data

![ROC_curve](images/roc_curve.png)
![PR_curve](images/pr_curve.png)


🧠 Probability Calibration & Business Threshold

![Bussiness Threeshold](images/bussiness_threesold.png)


Model probabilities are calibrated to represent realistic fraud risk scores

A custom threshold is applied instead of the default 0.5

This improves fraud detection based on business risk considerations

🧰 Technologies & Libraries Used

Python

NumPy

Pandas

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Seaborn

🚀 Key Learnings

Accuracy is not reliable for imbalanced datasets

Handling class imbalance is essential in fraud detection

SMOTE significantly improves minority class detection

Proper metrics and thresholds matter more than raw scores

Real-world ML requires balancing performance and system limits

🔮 Future Improvements

Try ensemble models (Random Forest, XGBoost)

Hyperparameter tuning

Time-aware validation

Real-time fraud detection API

Web deployment using Streamlit

✅ Conclusion

This project demonstrates a complete, real-world binary classification pipeline for fraud detection.
It emphasizes correct evaluation, imbalance handling, and engineering decisions, making it suitable for academic submission, GitHub portfolios, and interviews.
