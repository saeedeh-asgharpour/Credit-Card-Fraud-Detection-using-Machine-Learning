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

def load_dataset(path="credit_card.csv"):
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

