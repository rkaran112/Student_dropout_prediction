import streamlit as st
import plotly.express as px
import shap

from utils.preprocess import (
    build_single_input_dataframe,
    compute_risk_category,
    load_model_artifacts,
    prepare_features_from_df,
    recommendation_for_risk,
)


st.title("🧍 Individual Student Dropout Risk Predictor")

try:
    artifacts = load_model_artifacts()
    model = artifacts["model"]
except Exception:
    st.error(
        "Model not found. Please run the training script (model/train_model.py) before using this page."
    )
    st.stop()

with st.form("individual_predictor_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age at Enrollment", 17, 60, 20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
        fees_up_to_date = st.selectbox("Tuition Fees Up To Date", ["Yes", "No"])
        debtor = st.selectbox("Debtor", ["Yes", "No"])

    with col2:
        displaced = st.selectbox("Displaced Student", ["Yes", "No"])
        units_sem1_approved = st.slider("Units Approved (Sem 1)", 0, 26, 10)
        units_sem2_approved = st.slider("Units Approved (Sem 2)", 0, 26, 10)
        gpa_sem2 = st.slider("GPA - Semester 2 Grade", 0.0, 20.0, 12.0)
        unemployment_rate = st.slider("Unemployment Rate", 0.0, 25.0, 7.5)

    gdp = st.slider("GDP", 0.0, 100000.0, 20000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = build_single_input_dataframe(
        age=age,
        gender=gender,
        scholarship=scholarship,
        fees_up_to_date=fees_up_to_date,
        units_sem1_approved=units_sem1_approved,
        units_sem2_approved=units_sem2_approved,
        gpa_sem2=gpa_sem2,
        debtor=debtor,
        displaced=displaced,
        unemployment_rate=unemployment_rate,
        gdp=gdp,
    )

    X_processed_df, X_processed = prepare_features_from_df(input_df, artifacts)

    prob_dropout = float(model.predict_proba(X_processed)[0, 1])
    category = compute_risk_category(prob_dropout)
    recommendation = recommendation_for_risk(category)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dropout Risk Score", f"{prob_dropout * 100:.1f}%")
        st.metric("Risk Category", category)

    with col2:
        st.write("**Personalized Recommendation**")
        st.info(recommendation)

    st.markdown("---")

    st.subheader("🔍 Why this prediction?")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        # For binary classification, shap_values is a list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_vals_for_class = shap_values[1][0]
        else:
            shap_vals_for_class = shap_values[0]

        importances_df = (
            X_processed_df.iloc[0]
            .to_frame("value")
            .assign(shap_value=shap_vals_for_class)
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        importances_df["abs_shap"] = importances_df["shap_value"].abs()
        importances_df = importances_df.sort_values("abs_shap", ascending=False).head(10)

        fig = px.bar(
            importances_df,
            x="abs_shap",
            y="feature",
            orientation="h",
            title="Top Features Influencing This Prediction (|SHAP value|)",
            labels={"abs_shap": "|SHAP value|", "feature": "Feature"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Could not compute SHAP explanation. Showing only the prediction.")
