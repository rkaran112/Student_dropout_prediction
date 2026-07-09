import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.preprocess import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_single_input_dataframe,
    compute_risk_category,
    prepare_features_from_df,
    recommendation_for_risk,
)


def test_compute_risk_category_low():
    assert compute_risk_category(0.0) == "Low Risk"
    assert compute_risk_category(0.32) == "Low Risk"


def test_compute_risk_category_moderate():
    assert compute_risk_category(0.33) == "Moderate Risk"
    assert compute_risk_category(0.5) == "Moderate Risk"
    assert compute_risk_category(0.65) == "Moderate Risk"


def test_compute_risk_category_high():
    assert compute_risk_category(0.66) == "High Risk"
    assert compute_risk_category(1.0) == "High Risk"


def test_recommendation_for_risk_matches_category():
    assert "counseling" in recommendation_for_risk("High Risk").lower()
    assert "monitoring" in recommendation_for_risk("Moderate Risk").lower()
    assert "on track" in recommendation_for_risk("Low Risk").lower()


def test_build_single_input_dataframe_maps_yes_no_and_gender():
    df = build_single_input_dataframe(
        age=21,
        gender="Male",
        scholarship="Yes",
        fees_up_to_date="No",
        units_sem1_approved=5,
        units_sem2_approved=6,
        gpa_sem2=13.5,
        debtor="Yes",
        displaced="No",
        unemployment_rate=8.2,
        gdp=21000.0,
    )

    assert list(df.columns) == FEATURE_COLUMNS
    row = df.iloc[0]
    assert row["Age at enrollment"] == 21
    assert row["Gender"] == 1
    assert row["Scholarship holder"] == 1
    assert row["Tuition fees up to date"] == 0
    assert row["Debtor"] == 1
    assert row["Displaced"] == 0
    assert row["Curricular units 2nd sem (grade)"] == 13.5


def test_build_single_input_dataframe_unknown_values_fall_back_to_defaults():
    # Unrecognized category strings should not raise; they fall back to safe defaults.
    df = build_single_input_dataframe(
        age=19,
        gender="Unknown",
        scholarship="Unknown",
        fees_up_to_date="Unknown",
        units_sem1_approved=0,
        units_sem2_approved=0,
        gpa_sem2=0.0,
        debtor="Unknown",
        displaced="Unknown",
        unemployment_rate=0.0,
        gdp=0.0,
    )
    row = df.iloc[0]
    assert row["Gender"] == 0
    assert row["Scholarship holder"] == 0
    assert row["Tuition fees up to date"] == 1
    assert row["Debtor"] == 0
    assert row["Displaced"] == 0


def _fit_fake_artifacts():
    # Mirrors the shapes produced by model/train_model.py's preprocess(), but
    # fitted on a tiny in-memory frame so tests don't depend on the trained model file.
    raw = pd.DataFrame(
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
        }
    )

    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

    encoders = {}
    encoded = raw.copy()
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(raw[col])
        encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(raw[numeric_cols])

    artifacts = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": numeric_cols,
        "encoders": encoders,
        "scaler": scaler,
    }
    return raw, artifacts, numeric_cols


def test_prepare_features_from_df_encodes_and_scales():
    raw, artifacts, numeric_cols = _fit_fake_artifacts()

    X_df, X_values = prepare_features_from_df(raw, artifacts)

    assert list(X_df.columns) == FEATURE_COLUMNS
    assert (X_df.values == X_values).all()

    # Categorical columns should now hold the encoders' integer codes, not raw labels.
    for col in CATEGORICAL_COLUMNS:
        le = artifacts["encoders"][col]
        assert (X_df[col].values == le.transform(raw[col])).all()

    # Numeric columns should match the fitted scaler's own transform.
    expected_numeric = artifacts["scaler"].transform(raw[numeric_cols])
    assert (X_df[numeric_cols].values == expected_numeric).all()
