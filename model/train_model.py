import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]

# Allow running this file directly (`python model/train_model.py`), where only
# this file's own directory is added to sys.path by default.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.preprocess import CATEGORICAL_COLUMNS, FEATURE_COLUMNS  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download the UCI dataset, "
            "extract data.csv, rename it to dataset.csv and place it in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH, delimiter=";")
    if "Target" not in df.columns:
        raise ValueError("Expected 'Target' column in dataset.")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Binary target: 1 = Dropout, 0 = Non-Dropout
    y = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0).values

    # Select features
    X = df[FEATURE_COLUMNS].copy()

    # Label-encode categorical columns
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

    # Scale numeric columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, encoders, scaler, numeric_cols


def train():
    df = load_data()
    X, y, encoders, scaler, numeric_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Handle class imbalance with SMOTE on the training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    report = classification_report(y_test, y_pred, target_names=["Non-Dropout", "Dropout"])

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_distribution": {
            "train_positive": int(np.sum(y_train)),
            "train_negative": int(len(y_train) - np.sum(y_train)),
            "test_positive": int(np.sum(y_test)),
            "test_negative": int(len(y_test) - np.sum(y_test)),
        },
    }

    artifacts = {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": numeric_cols,
        "metrics": metrics,
    }

    joblib.dump(artifacts, MODEL_DIR / "dropout_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Model Evaluation Metrics (Test Set)\n")
        f.write("=" * 40 + "\n\n")
        f.write(report + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")

    print("Training completed.")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    train()
