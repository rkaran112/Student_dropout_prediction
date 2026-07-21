import sys
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _metrics_by_label(at):
    return {m.label: m.value for m in at.metric}


def test_home_page_renders_summary_metrics():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "🎓 Smart Student Dropout Risk Predictor"

    metrics = _metrics_by_label(at)
    assert metrics["Total Students in Dataset"] == "4424"
    assert metrics["Observed Dropout Rate"] == "32.1%"
    assert metrics["Model Accuracy (Test)"] == "86.8%"


def test_home_page_shows_warning_when_model_missing():
    at = AppTest.from_file("app.py")
    with mock.patch(
        "utils.preprocess.load_model_artifacts", side_effect=FileNotFoundError("no model")
    ):
        at.run(timeout=30)

    assert not at.exception
    assert any("Model artifacts not found" in w.value for w in at.warning)
    assert _metrics_by_label(at)["Model Accuracy (Test)"] == "N/A"


def test_home_page_handles_unreadable_dataset():
    at = AppTest.from_file("app.py")
    with mock.patch("pandas.read_csv", side_effect=Exception("bad csv")):
        at.run(timeout=30)

    assert not at.exception
    metrics = _metrics_by_label(at)
    assert metrics["Total Students in Dataset"] == "N/A"
    assert metrics["Observed Dropout Rate"] == "N/A"
