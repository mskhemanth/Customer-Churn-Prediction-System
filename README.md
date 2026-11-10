# Customer-Churn-Prediction-System
## 📌 Overview
An end-to-end Machine Learning pipeline developed in Python to predict customer churn and identify high-risk accounts. Designed with statistical rigor, exploratory data analysis, class imbalance mitigation, and cross-model performance benchmarking.

---

## 🛠 Tech Stack
- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib
- **Concepts:** Supervised Classification (Logistic Regression, Random Forest), Feature Scaling, One-Hot Encoding, Cross-Validation, ROC-AUC Analysis

---

## 🚀 Key Steps & Architecture
1. **Data Ingestion & Cleaning:**
   - Handled categorical variables and numerical feature distributions.
   - Checked for missing records, anomalies, and multi-collinearity.
2. **Exploratory Data Analysis (EDA):**
   - Evaluated churn correlations with contract duration, support escalations, and monthly charges.
3. **Pipeline Construction:**
   - Unified preprocessing with Scikit-learn's `ColumnTransformer` (standardization for numerical features, dummy encoding for categorical features).
4. **Model Training & Evaluation:**
   - Trained **Logistic Regression** and **Random Forest Classifier** with balanced class weighting.
   - Assessed accuracy using **Confusion Matrix, Precision, Recall, F1-Score**, and **ROC-AUC** to balance false positives vs. false negatives.

---

## 📁 Repository Structure
```
├── customer_churn_data.csv       # Sample structured dataset
├── train_churn_model.py         # End-to-end ML pipeline script
└── README.md                    # Project documentation
```

---

## ⚡ How to Run
```bash
pip install pandas scikit-learn numpy matplotlib
python train_churn_model.py
```
