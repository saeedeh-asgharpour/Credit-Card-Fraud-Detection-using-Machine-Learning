"""
End-to-End Credit Card Fraud Detection Pipeline
Handles class imbalance, preprocessing, model training, evaluation,
and complete ML workflow with professional standards.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_dataset(path="creditcard.csv"):
    df = pd.read_csv(path)
    return df

def dataset_analysis(df):
    fraud = df[df["Class"] == 1]
    valid = df[df["Class"] == 0]

    print("=== Dataset Overview ===")
    print(f"Total Transactions: {len(df)}")
    print(f"Fraudulent: {len(fraud)}")
    print(f"Valid: {len(valid)}")
    print(f"Fraud Ratio: {len(fraud) / len(df):.6f}\n")

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def preprocess(df):
    x = df.drop("Class", axis = 1)
    y = df["Class"]

    # Feature Scaling
    scaler = StandardScaler()
    x[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
    return x, y, scaler

def build_pipeline():
    return Pipeline([
        ("undersample", RandomUnderSampler(sampling_strategy=0.8)),
        ("smote", SMOTE(sampling_strategy=0.1)),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

def tune_model(pipeline, X, y):
    params = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y)
    print("Best Parameters:", grid.best_params_)
    print("Best ROC-AUC Score:", grid.best_score_)
    return grid.best_estimator_
# ===========================================================
# Evaluation
# ===========================================================
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC:   {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC:    {pr_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["Valid", "Fraud"],
                yticklabels=["Valid", "Fraud"])
    plt.title("Confusion Matrix")
    plt.show()
    # PR Curve
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()


# ===========================================================
# MAIN WORKFLOW
# ===========================================================
def main():
    df = load_dataset()
    dataset_analysis(df)

    X, y, scaler = preprocess(df)

    # Preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    print("\nRunning Hyperparameter Tuning...")
    best_model = tune_model(pipeline, X_train, y_train)
    print("\nEvaluating Final Model...")
    evaluate(best_model, X_test, y_test)


if __name__ == "__main__":
    main()