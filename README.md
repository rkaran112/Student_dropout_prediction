# Student Dropout Prediction

A Streamlit web app that predicts student dropout risk from academic, financial, and demographic indicators, using a Random Forest model trained on the UCI "Predict Students' Dropout and Academic Success" dataset.

## What It Does

The app helps faculty, administrators, and counselors identify students who may be at risk of dropping out. It provides:

- **Individual Predictor** (`pages/1_Individual_Predictor.py`) — enter a single student's details and get a dropout risk score with a SHAP-based explanation and a recommended action.
- **Batch Prediction** (`pages/2_Batch_Prediction.py`) — upload a CSV of multiple students and get predictions for all of them.
- **Analytics Dashboard** (`pages/3_Analytics_Dashboard.py`) — visual exploration of the dataset (dropout rates, distributions, etc.).
- **Model Info** (`pages/4_Model_Info.py`) — view model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).

The home page (`app.py`) shows summary stats (dataset size, observed dropout rate, model accuracy) and links to the pages above.

## Tech Stack

- **Python 3.11** (see `runtime.txt`)
- **Streamlit** — web UI / multipage app
- **scikit-learn** (`RandomForestClassifier`) — classification model
- **imbalanced-learn** (SMOTE) — handles class imbalance in training
- **pandas / numpy** — data handling
- **plotly** — visualizations
- **shap** — model explainability
- **joblib** — model/scaler persistence

Full pinned dependency list in `requirements.txt`.

## Setup / Install

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset: get the "Predict Students' Dropout and Academic Success" dataset ZIP from the UCI Machine Learning Repository, extract `data.csv`, rename it to `dataset.csv`, and place it in `data/dataset.csv` (semicolon-delimited). See `data/README.md` for details.
3. Train the model (produces `model/dropout_model.pkl`, `model/scaler.pkl`, `model/metrics.txt`):
   ```bash
   python model/train_model.py
   ```

A pre-trained model and dataset are already committed to this repo (`model/dropout_model.pkl`, `data/dataset.csv`), so the app can also be run directly without retraining.

## Usage

Run the Streamlit app from the repo root:

```bash
streamlit run app.py
```

Then use the sidebar to navigate between the Individual Predictor, Batch Prediction, Analytics Dashboard, and Model Info pages.

## Model Performance

From `model/metrics.txt` (Random Forest, test set):

| Metric | Value |
|---|---|
| Accuracy | 0.868 |
| Precision (Dropout) | 0.815 |
| Recall (Dropout) | 0.761 |
| F1-score (Dropout) | 0.787 |
| ROC-AUC | 0.903 |

## Status

**Complete / working end-to-end.** The training pipeline, saved model artifacts, and all four Streamlit pages are implemented (no stub functions or TODOs found in the source). Not independently verified by running the app in this pass — review the code before deploying to production. Unit tests for the `utils/preprocess.py` helpers live in `tests/test_preprocess.py` (run with `pytest`); the Streamlit pages themselves are not covered by automated tests.
