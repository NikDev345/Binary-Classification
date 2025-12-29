💳 Credit Card Fraud Detection
Binary Classification using Machine Learning
📌 Project Overview

Credit card fraud is a serious financial problem where fraudulent transactions cause significant losses to both customers and banks.
This project focuses on building a binary classification machine learning model that can identify whether a credit card transaction is fraudulent or legitimate.

The project demonstrates how machine learning models behave on highly imbalanced real-world datasets and why accuracy alone is not a reliable metric.

🎯 Problem Statement

Given a dataset of credit card transactions, the goal is to classify each transaction into one of two classes:

0 → Legitimate Transaction

1 → Fraudulent Transaction

This is a binary classification problem.

📊 Dataset Information

Dataset Name: Credit Card Fraud Detection

Source: Kaggle

File Name: creditcard.csv

Total Transactions: ~284,000

Fraud Cases: ~492 (highly imbalanced dataset)

🔹 The dataset contains anonymized features (V1 to V28) generated using PCA, along with:

Amount → Transaction amount

Class → Target label (0 or 1)

📥 Dataset Download Instructions

Visit Kaggle

Search for “Credit Card Fraud Detection”

Download the dataset

Place the file as:

data/creditcard.csv

🧠 Machine Learning Approach
✔ Type of Learning

Supervised Learning

✔ Problem Type

Binary Classification

✔ Model Used

Logistic Regression (baseline model)

✔ Key Challenge

Severe class imbalance

Fraud transactions are extremely rare compared to legitimate ones

⚙️ Project Workflow

Load and inspect the dataset

Analyze class imbalance

Split features and target labels

Perform stratified train–test split

Train a baseline Logistic Regression model

Evaluate the model using multiple metrics

Handle class imbalance using SMOTE

Retrain the model and compare results

📈 Evaluation Metrics

Accuracy alone is misleading for imbalanced datasets.
Therefore, the following metrics are used:

Precision

Recall

F1-Score

Confusion Matrix

🔹 Recall is prioritized, because missing a fraudulent transaction is more costly than incorrectly flagging a legitimate one.

🧪 Handling Class Imbalance

To improve fraud detection performance, SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by generating synthetic fraud samples.

This significantly improves the model’s ability to detect fraudulent transactions.

🧰 Technologies & Libraries Used

Python

NumPy

Pandas

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Seaborn

📊 Model Performance Visualization

To better understand the impact of handling class imbalance, confusion matrix visualizations are included before and after applying SMOTE.

🔹 Before SMOTE (Imbalanced Dataset)

This confusion matrix shows the model performance on the original, highly imbalanced dataset.
It highlights how the model struggles to correctly identify fraudulent transactions due to class imbalance.

🔹 After SMOTE (Balanced Dataset)

After applying SMOTE (Synthetic Minority Over-sampling Technique), the dataset becomes balanced.
This confusion matrix demonstrates a significant improvement in recall for fraudulent transactions.

🧠 Key Observation

Before SMOTE:
High accuracy but very poor fraud detection (low recall)

After SMOTE:
Improved fraud detection with better recall and balanced learning

This comparison clearly shows why accuracy alone is misleading for imbalanced binary classification problems.


![Before SMOTE](figure1.png)
![After SMOTE](figure2.png)
