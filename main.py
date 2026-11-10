"""
Customer Churn Prediction Model
Tech Stack: Python, Pandas, NumPy, Scikit-Learn
Purpose: End-to-end data pipeline, exploratory analysis, preprocessing, and classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

def run_pipeline():
    print("1. Loading dataset...")
    df = pd.read_csv("customer_churn_data.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean() * 100:.2f}%\n")

    # Features and target
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]

    num_features = ["TenureMonths", "MonthlyCharges", "TotalCharges", "SupportTickets"]
    cat_features = ["ContractType"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
    }

    print("2. Training and Evaluating Models...")
    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        print(f"=== {name} ===")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 40)

if __name__ == "__main__":
    run_pipeline()
