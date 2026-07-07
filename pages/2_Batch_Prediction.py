import io

import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocess import (
    compute_risk_category,
    load_model_artifacts,
    prepare_features_from_df,
    recommendation_for_risk,
)


st.title("📂 Batch Prediction (CSV Upload)")

try:
    artifacts = load_model_artifacts()
    model = artifacts["model"]
except Exception:
    st.error(
        "Model not found. Please run the training script (model/train_model.py) before using this page."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # UCI dataset and recommended sample files are semicolon-separated
        df_raw = pd.read_csv(uploaded_file, delimiter=";")
    except Exception:
        st.error("Could not read the uploaded file. Ensure it is a valid CSV.")
        st.stop()

    feature_cols = artifacts["feature_columns"]
    missing_cols = [c for c in feature_cols if c not in df_raw.columns]

    if missing_cols:
        st.error(
            "The uploaded file is missing required columns: " + ", ".join(missing_cols)
        )
        st.stop()

    try:
        X_processed_df, X_processed = prepare_features_from_df(df_raw, artifacts)
        probs = model.predict_proba(X_processed)[:, 1]
    except ValueError as e:
        st.error(
            "Could not process the uploaded data. Please check that categorical and "
            f"numeric columns contain only values seen during training.\n\nDetails: {e}"
        )
        st.stop()

    risk_labels = [compute_risk_category(p) for p in probs]
    recommendations = [recommendation_for_risk(c) for c in risk_labels]

    result_df = df_raw.copy()
    result_df["Risk_Score"] = (probs * 100).round(1)
    result_df["Risk_Label"] = risk_labels
    result_df["Recommendation"] = recommendations

    st.subheader("Prediction Results")

    def highlight_row(row):
        color = ""
        if row["Risk_Label"] == "High Risk":
            color = "background-color: #ffcccc"  # light red
        elif row["Risk_Label"] == "Moderate Risk":
            color = "background-color: #fff3cd"  # light yellow
        else:
            color = "background-color: #d4edda"  # light green
        return [color] * len(row)

    styled_df = result_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")

    total_students = len(result_df)
    high_risk = np.sum(result_df["Risk_Label"] == "High Risk")
    moderate_risk = np.sum(result_df["Risk_Label"] == "Moderate Risk")
    low_risk = np.sum(result_df["Risk_Label"] == "Low Risk")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", total_students)
    with col2:
        st.metric("High Risk", high_risk)
    with col3:
        st.metric("Moderate Risk", moderate_risk)
    with col4:
        st.metric("Low Risk", low_risk)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="batch_predictions_with_risk.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV file to run batch predictions.")
