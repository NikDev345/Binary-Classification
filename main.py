import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from bert_nlp import get_bert_embeddings

# Load data
df = pd.read_csv("data.csv")
# columns: ["text", "label"]

X = df["text"].tolist()
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# BERT embeddings
X_train_emb = get_bert_embeddings(X_train).numpy()
X_test_emb = get_bert_embeddings(X_test).numpy()

# Classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_emb, y_train)

# Evaluation
y_pred = model.predict(X_test_emb)
print(classification_report(y_test, y_pred))
