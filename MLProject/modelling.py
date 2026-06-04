import os
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

TARGET_COL = "HeartDisease"
EXPERIMENT_NAME = "heartdisease-baseline-local"
RUN_NAME = "logreg"
TEST_SIZE = 0.2
RANDOM_STATE = 42

DATASET_PATH = Path(__file__).parent / "dataset_preprocessed.csv"

if os.environ.get("GITHUB_ACTIONS") == "true":
    MLFLOW_TRACKING_URI = "file://" + str(Path(__file__).parent.parent / "mlruns")
else:
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def load_xy(dataset_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(dataset_path)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    return metrics


def main() -> int:
    # Validasi keberadaan file dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di jalur: {DATASET_PATH.resolve()}"
        )

    # Inisialisasi MLflow dengan alamat localhost
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dan split data
    x, y = load_xy(DATASET_PATH, TARGET_COL)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Inisialisasi Model Baseline
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    # Aktifkan MLflow autolog
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, silent=True)

    # Mulai proses training dan logging ke MLflow
    with mlflow.start_run(run_name=RUN_NAME):
        model.fit(x_train, y_train)

        # Prediksi hasil
        y_pred = model.predict(x_test)

        # Hitung dan catat metrics evaluasi
        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Log model secara manual beserta signature data
        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_manual_copy",
            signature=signature,
            input_example=x_train.head(5),
        )

        print("-" * 50)
        print("MLflow Tracking URI :", mlflow.get_tracking_uri())
        print("Nama Eksperimen     :", EXPERIMENT_NAME)
        print("Hasil Metrics       :", metrics)
        print("-" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
