import streamlit as st
import pandas as pd
from pathlib import Path

from utils.preprocess import load_model_artifacts

st.set_page_config(
    page_title="Smart Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Smart Student Dropout Risk Predictor")
st.subheader("CHRIST (Deemed to be University)")

st.markdown(
    """
This application helps **faculty, administrators, and counselors** identify students who may be
at risk of dropping out, based on academic, financial, and behavioral indicators.

Use the pages in the sidebar to:
- Predict **individual student** dropout risk
- Run **batch predictions** from CSV files
- Explore **analytics dashboards** and **model details**
    """
)

base_dir = Path(__file__).resolve().parent

def load_dataset_summary():
    data_path = base_dir / "data" / "dataset.csv"
    if not data_path.exists():
        return None
    try:
        df = pd.read_csv(data_path, delimiter=";")
        total_students = len(df)
        dropout_rate = None
        if "Target" in df.columns:
            dropout_binary = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)
            dropout_rate = dropout_binary.mean()
        return {
            "total_students": total_students,
            "dropout_rate": dropout_rate,
        }
    except Exception:
        return None

artifacts = None
metrics = None
try:
    artifacts = load_model_artifacts()
    metrics = artifacts.get("metrics", {}) if isinstance(artifacts, dict) else {}
except Exception:
    st.warning(
        "Model artifacts not found. Please run the training script in the **model/train_model.py** file first."
    )

col1, col2, col3 = st.columns(3)

summary = load_dataset_summary()

with col1:
    if summary is not None:
        st.metric("Total Students in Dataset", f"{summary['total_students']}")
    else:
        st.metric("Total Students in Dataset", "N/A")

with col2:
    if summary is not None and summary.get("dropout_rate") is not None:
        st.metric(
            "Observed Dropout Rate",
            f"{summary['dropout_rate'] * 100:.1f}%",
        )
    else:
        st.metric("Observed Dropout Rate", "N/A")

with col3:
    acc = metrics.get("accuracy") if metrics else None
    if acc is not None:
        st.metric("Model Accuracy (Test)", f"{acc * 100:.1f}%")
    else:
        st.metric("Model Accuracy (Test)", "N/A")

st.markdown("---")

st.markdown(
    """
### 📊 How to Use This App

- Go to **"1_Individual_Predictor"** to test a single student's risk
- Go to **"2_Batch_Prediction"** to upload a CSV of many students
- Explore **"3_Analytics_Dashboard"** for visual insights
- See **"4_Model_Info"** for model performance and details

Make sure you have downloaded the UCI dataset, saved it as `data/dataset.csv`,
and run `python model/train_model.py` at least once to train and save the model.
    """
)
