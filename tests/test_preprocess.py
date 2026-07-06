import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.preprocess import (
    FEATURE_COLUMNS,
    build_single_input_dataframe,
    compute_risk_category,
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
