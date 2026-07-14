import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from utils.preprocess import load_model_artifacts


st.title("🧠 Model Information & Performance")

try:
    artifacts = load_model_artifacts()
except Exception:
    st.error(
        "Model artifacts not found. Please run the training script (model/train_model.py) first."
    )
    st.stop()

metrics = artifacts.get("metrics", {})
model = artifacts.get("model")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
with col2:
    st.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
with col3:
    st.metric("Recall", f"{metrics.get('recall', 0) * 100:.2f}%")
with col4:
    st.metric("F1-score", f"{metrics.get('f1', 0) * 100:.2f}%")

st.markdown("---")

st.subheader("Confusion Matrix (Test Set)")

cm = metrics.get("confusion_matrix")
if cm is not None:
    cm = np.array(cm)
    z = cm
    x = ["Predicted Non-Dropout", "Predicted Dropout"]
    y = ["Actual Non-Dropout", "Actual Dropout"]

    fig_cm = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)
else:
    st.info("Confusion matrix not available.")

st.markdown("---")

st.subheader("ROC Curve (Test Set)")

fpr = metrics.get("fpr")
tpr = metrics.get("tpr")
roc_auc = metrics.get("roc_auc")

if fpr is not None and tpr is not None:
    fig_roc = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC = {roc_auc:.3f})" if roc_auc is not None else "ROC Curve",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    )
    fig_roc.add_shape(
        type="line",
        line=dict(dash="dash"),
        x0=0,
        x1=1,
        y0=0,
        y1=1,
    )
    st.plotly_chart(fig_roc, use_container_width=True)
else:
    st.info("ROC data not available.")

st.markdown("---")

st.subheader("Model Details")

st.write("**Model Type:** RandomForestClassifier")

if model is not None:
    st.write(f"Number of trees: {getattr(model, 'n_estimators', 'N/A')}")
    st.write(f"Max depth: {getattr(model, 'max_depth', 'N/A')}")

st.markdown(
    """
### Preprocessing Steps

- Encoded categorical variables using **LabelEncoder**
- Scaled numeric variables using **StandardScaler**
- Handled class imbalance using **SMOTE** on the training set

This model predicts a binary outcome: **Dropout (1)** vs **Non-Dropout (0)** based on
student demographics, academic performance, and socio-economic indicators.
    """
)
