# 📌 Credit Card Fraud Detection — Machine Learning Pipeline

## 📖 Overview

This project implements a complete **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions using the popular **Credit Card Fraud Dataset**.
Fraud detection is a highly imbalanced classification problem, and this project applies proper preprocessing, resampling, modeling, and evaluation techniques to achieve strong performance.

The workflow includes:

* Data loading & exploration
* Handling imbalanced classes
* Feature scaling
* Train/test split
* Random Forest classification
* Evaluation using industry-standard metrics
* Visualization of correlation matrix & confusion matrix

---

## 📂 Dataset

The dataset used in this project is the **Kaggle Credit Card Fraud Detection Dataset**, originally from a European card issuer.

* Contains **284,807** transactions
* Only **492 (0.172%)** are fraudulent
* Features `V1–V28` are PCA-transformed for anonymity
* `Amount` and `Time` are not transformed
* Target variable:

  * `0` → Normal transaction
  * `1` → Fraudulent transaction

You must place the dataset file in the project directory:

```
creditcard.csv
```

---

## ⚙️ Installation & Requirements

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

---

## 🚀 How to Run

```bash
python fraud_detection.py
```

Replace `fraud_detection.py` with your script filename.

---

## 🧠 Model Pipeline Summary

### 1. **Data Exploration**

* Calculates fraud ratio
* Prints statistical summaries
* Displays correlation heatmap

### 2. **Preprocessing**

* Splits data into input features `X` and target `y`
* Scales numerical columns (`Time`, `Amount`) using `StandardScaler`
* Stratified train/test split to preserve class imbalance

### 3. **Model**

The model used is:

```python
RandomForestClassifier(
    class_weight="balanced",
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

`class_weight="balanced"` helps compensate for severe class imbalance.

### 4. **Evaluation**

Metrics include:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Matthews Correlation Coefficient (MCC)**
* **Confusion Matrix**
* **ROC-AUC / PR-AUC (optional)**

Visualizations include:

* Correlation heatmap
* Confusion matrix heatmap

---

## 📊 Example Results

Typical performance for Random Forest on this dataset:

| Metric    | Expected Range              |
| --------- | --------------------------- |
| Accuracy  | 0.99+                       |
| Precision | 0.90–0.98                   |
| Recall    | 0.70–0.95 (critical metric) |
| F1-Score  | 0.80–0.96                   |
| MCC       | 0.80–0.95                   |

Actual values depend on train/test split and random seed.

---

## 📁 Project Structure

```
📦 credit-card-fraud-detection
│
├── fraud_detection.py       # Your main code file
├── creditcard.csv           # Dataset (not included)
├── README.md                # Project documentation
└── requirements.txt         # Optional dependency list
```

---

## 🧩 Possible Improvements (Future Work)

You can extend this project by adding:

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Advanced models (XGBoost, LightGBM, CatBoost)
* SMOTE and hybrid resampling techniques
* Feature engineering and anomaly detection approaches
* Deployment with FastAPI / Flask
* Model interpretability (SHAP values)

---
