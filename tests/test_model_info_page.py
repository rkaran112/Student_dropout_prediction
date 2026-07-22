import sys
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _metrics_by_label(at):
    return {m.label: m.value for m in at.metric}


def test_model_info_page_renders_metrics():
    at = AppTest.from_file("pages/4_Model_Info.py")
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "🧠 Model Information & Performance"

    metrics = _metrics_by_label(at)
    assert metrics["Accuracy"] == "86.78%"
    assert metrics["Precision"] == "81.51%"
    assert metrics["Recall"] == "76.06%"
    assert metrics["F1-score"] == "78.69%"


def test_model_info_page_shows_error_when_model_missing():
    at = AppTest.from_file("pages/4_Model_Info.py")
    with mock.patch(
        "utils.preprocess.load_model_artifacts", side_effect=FileNotFoundError("no model")
    ):
        at.run(timeout=30)

    assert not at.exception
    assert any("Model artifacts not found" in e.value for e in at.error)
