import io
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(__file__))
from feature_utils import RAW_MODEL_FEATURES, engineer_features, validate_raw_features

st.set_page_config(page_title="Credit Card Default Risk", layout="wide")

HF_USERNAME = os.getenv("HF_USERNAME")
MODEL_REPO = f"{HF_USERNAME}/credit-card-default-model"
TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_assets():
    model_path = hf_hub_download(MODEL_REPO, "credit_default_model.joblib", token=TOKEN)
    metadata_path = hf_hub_download(MODEL_REPO, "model_metadata.json", token=TOKEN)
    return joblib.load(model_path), json.load(open(metadata_path, encoding="utf-8"))

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error(f"Model could not be loaded. Check HF_USERNAME and repository access. Details: {exc}")
    st.stop()

threshold = float(metadata["threshold"])
status_options = {
    "No consumption (-2)": -2,
    "Paid duly (-1)": -1,
    "Revolving credit use (0)": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3 months delay": 3,
    "4 months delay": 4,
    "5 months delay": 5,
    "6 months delay": 6,
    "7 months delay": 7,
    "8 months delay": 8,
    "9+ months delay": 9,
}
months = [
    ("September 2005", "PAY_0", "BILL_AMT1", "PAY_AMT1"),
    ("August 2005", "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("July 2005", "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("June 2005", "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("May 2005", "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("April 2005", "PAY_6", "BILL_AMT6", "PAY_AMT6"),
]

st.title("Credit Card Default Risk Prediction")
st.caption("Decision-support demonstration. The model excludes ID, gender, education and marital status from prediction.")

overview_tab, single_tab, batch_tab = st.tabs(["Overview", "Single Customer", "Batch Upload"])

with overview_tab:
    st.subheader("Application overview")
    st.write("The model estimates next-month default probability using credit limit, age, six-month repayment status, bill history and payment history.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", metadata["model_name"])
    c2.metric("Decision threshold", f"{threshold:.2f}")
    c3.metric("Model inputs", len(RAW_MODEL_FEATURES))
    st.info("Risk bands are communication aids: Low < 30%, Medium 30–60%, High > 60%. They are not official banking policy thresholds.")

with single_tab:
    st.subheader("Single-customer assessment")
    with st.form("single_customer_form"):
        c1, c2 = st.columns(2)
        limit_bal = c1.number_input("Credit limit (NT$)", min_value=1.0, value=200000.0, step=10000.0)
        age = c2.number_input("Age", min_value=18, max_value=100, value=35)

        st.markdown("#### Six-month repayment status")
        statuses = {}
        status_cols = st.columns(3)
        for i, (month, pay_col, _, _) in enumerate(months):
            label = status_cols[i % 3].selectbox(month, list(status_options), index=2, key=f"status_{pay_col}")
            statuses[pay_col] = status_options[label]

        st.markdown("#### Six-month bill history (NT$)")
        bills = {}
        bill_cols = st.columns(3)
        for i, (month, _, bill_col, _) in enumerate(months):
            bills[bill_col] = bill_cols[i % 3].number_input(month, value=50000.0, step=1000.0, key=f"bill_{bill_col}")

        st.markdown("#### Six-month payment history (NT$)")
        payments = {}
        payment_cols = st.columns(3)
        for i, (month, _, _, payment_col) in enumerate(months):
            payments[payment_col] = payment_cols[i % 3].number_input(month, min_value=0.0, value=5000.0, step=500.0, key=f"payment_{payment_col}")

        submitted = st.form_submit_button("Estimate default risk")

    if submitted:
        record = {"LIMIT_BAL": limit_bal, "AGE": age, **statuses, **bills, **payments}
        raw = pd.DataFrame([record])
        probability = float(model.predict_proba(engineer_features(raw))[:, 1][0])
        prediction = int(probability >= threshold)
        risk = "Low" if probability < 0.30 else "Medium" if probability <= 0.60 else "High"
        st.metric("Default probability", f"{probability:.1%}")
        st.write(f"**Risk band:** {risk}")
        if prediction:
            st.warning("Model result: Flag for additional credit-risk review.")
        else:
            st.success("Model result: Not flagged at the selected threshold.")

with batch_tab:
    st.subheader("Batch scoring")
    st.write("Upload a CSV containing the required raw model columns. Extra columns such as ID may be included and will be preserved in the output.")
    template = pd.DataFrame(columns=RAW_MODEL_FEATURES)
    st.download_button("Download input template", template.to_csv(index=False), "batch_input_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload customer CSV", type="csv")

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            validate_raw_features(batch)
            probabilities = model.predict_proba(engineer_features(batch))[:, 1]
            scored = batch.copy()
            scored["default_probability"] = probabilities
            scored["predicted_default"] = (probabilities >= threshold).astype(int)
            scored["risk_band"] = pd.cut(
                probabilities, bins=[-0.001, 0.30, 0.60, 1.0],
                labels=["Low", "Medium", "High"]
            ).astype(str)
            st.success(f"Scored {len(scored):,} customers.")
            st.dataframe(scored.head(50), use_container_width=True)
            st.download_button(
                "Download scored file", scored.to_csv(index=False),
                "credit_default_scored.csv", "text/csv"
            )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")
