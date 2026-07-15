import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model.train_model as train_model
from model.train_model import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_data, preprocess


def _raw_dataframe():
    return pd.DataFrame(
        {
            "Age at enrollment": [18, 22, 30, 45],
            "Gender": [0, 1, 0, 1],
            "Scholarship holder": [0, 1, 0, 1],
            "Tuition fees up to date": [1, 0, 1, 0],
            "Curricular units 1st sem (approved)": [3, 6, 0, 8],
            "Curricular units 2nd sem (approved)": [2, 6, 0, 8],
            "Curricular units 2nd sem (grade)": [10.0, 15.5, 0.0, 18.0],
            "Debtor": [0, 1, 0, 1],
            "Displaced": [1, 0, 1, 0],
            "Unemployment rate": [7.5, 10.2, 5.1, 8.8],
            "GDP": [19000.0, 21500.0, 17000.0, 23000.0],
            "Target": ["Dropout", "Graduate", "Dropout", "Enrolled"],
        }
    )


def test_load_data_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(train_model, "DATA_PATH", tmp_path / "does_not_exist.csv")

    with pytest.raises(FileNotFoundError, match="Download the UCI dataset"):
        load_data()


def test_load_data_raises_when_target_column_missing(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"Age at enrollment": [18, 22]}).to_csv(csv_path, sep=";", index=False)
    monkeypatch.setattr(train_model, "DATA_PATH", csv_path)

    with pytest.raises(ValueError, match="Expected 'Target' column"):
        load_data()


def test_load_data_reads_semicolon_delimited_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    _raw_dataframe().to_csv(csv_path, sep=";", index=False)
    monkeypatch.setattr(train_model, "DATA_PATH", csv_path)

    df = load_data()

    assert "Target" in df.columns
    assert len(df) == 4


def test_preprocess_binarizes_dropout_target():
    raw = _raw_dataframe()

    _, y, _, _, _ = preprocess(raw)

    assert list(y) == [1, 0, 1, 0]


def test_preprocess_encodes_and_scales_features():
    raw = _raw_dataframe()

    X, _, encoders, scaler, numeric_cols = preprocess(raw)

    assert list(X.columns) == FEATURE_COLUMNS
    assert set(encoders.keys()) == set(CATEGORICAL_COLUMNS)
    assert set(numeric_cols) == set(FEATURE_COLUMNS) - set(CATEGORICAL_COLUMNS)

    # Categorical columns should hold integer codes from the fitted encoders.
    for col in CATEGORICAL_COLUMNS:
        assert (X[col].values == encoders[col].transform(raw[col])).all()

    # Numeric columns should be scaled to zero mean via the fitted StandardScaler.
    assert X[numeric_cols].values.mean(axis=0) == pytest.approx(0.0, abs=1e-8)
    assert (X[numeric_cols].values == scaler.transform(raw[numeric_cols])).all()
