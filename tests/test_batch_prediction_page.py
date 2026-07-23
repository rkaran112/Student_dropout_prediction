import sys
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_batch_prediction_page_prompts_for_upload_by_default():
    at = AppTest.from_file("pages/2_Batch_Prediction.py")
    at.run(timeout=30)

    assert not at.exception
    assert at.title[0].value == "📂 Batch Prediction (CSV Upload)"
    assert any("Upload a CSV file" in i.value for i in at.info)


def test_batch_prediction_page_shows_error_when_model_missing():
    at = AppTest.from_file("pages/2_Batch_Prediction.py")
    with mock.patch(
        "utils.preprocess.load_model_artifacts", side_effect=FileNotFoundError("no model")
    ):
        at.run(timeout=30)

    assert not at.exception
    assert any("Model not found" in e.value for e in at.error)
