import os
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
from src.data_processing import build_feature_engineering_pipeline
from src.model_utils import calculate_credit_score, predict_optimal_loan
from src.explainability import generate_shap_explanations

st.set_page_config(page_title="Bati Bank Credit Risk Dashboard", layout="wide")
st.title("Bati Bank Credit Risk Dashboard")

MODEL_OPTIONS = [
    "CreditRiskModel_LogisticRegression",
    "CreditRiskModel_RandomForest"
]

model_name = st.selectbox("Model", MODEL_OPTIONS, index=0)
model_stage = st.selectbox("Stage", ["Production", "Staging"], index=0)


@st.cache_resource(show_spinner=False)
def load_model(name: str, stage: str):
    uri = f"models:/{name}/{stage}"
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_model(model_name, model_stage)

uploaded = st.file_uploader("Upload CSV with raw transactions", type=["csv"])

if uploaded and model is not None:
    df = pd.read_csv(uploaded)
    st.write("Raw shape:", df.shape)

    # We build a FE pipeline based on the uploaded data schema (without target)
    tmp_df = df.copy()
    # placeholder to satisfy WOE target presence during build
    tmp_df['is_high_risk'] = 0
    fe = build_feature_engineering_pipeline(tmp_df, use_woe_iv=True)

    X_fe = fe.fit_transform(tmp_df, tmp_df['is_high_risk'])
    st.write("Processed shape:", getattr(X_fe, 'shape', None))

    # Predict probabilities using full sklearn pipeline model
    try:
        probs = model.predict_proba(df)[:, 1]
    except Exception:
        preds = model.predict(df)
        probs = preds if preds.ndim == 1 else preds[:, -1]

    scores = calculate_credit_score(probs)
    avg_prob = float(probs.mean())
    avg_score = float(scores.mean())
    avg_amount, avg_duration = predict_optimal_loan(avg_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Risk Probability", f"{avg_prob:.2%}")
    c2.metric("Avg Credit Score", f"{avg_score:.0f}")
    c3.metric("Loan Suggestion", f"${avg_amount:,.2f} / {avg_duration} mo")

    st.subheader("SHAP Summary")
    try:
        img_path = generate_shap_explanations(
            model, df.sample(min(100, len(df))), df.iloc[:50])
        st.image(img_path, caption="SHAP Summary", use_column_width=True)
    except Exception as e:
        st.warning(f"SHAP failed: {e}")
else:
    st.info("Upload a CSV to see predictions and explanations.")
