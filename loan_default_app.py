
# stream.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("loan_default_model.pkl")
preprocessor = joblib.load("loan_preprocessor.pkl")

st.title("Loan Default Prediction Explorer")

# Load results (or generate again)
@st.cache_data
def get_data():
    np.random.seed(42)
    n = 2000
    age = np.random.randint(18, 65, size=n)
    income = np.random.normal(2500, 1000, size=n).clip(500, None)
    loan_amount = np.random.normal(10000, 5000, size=n).clip(1000, None)
    employment_type = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], size=n, p=[0.6, 0.3, 0.1])
    credit_score = np.random.normal(650, 50, size=n).clip(300, 850)
    tenure = np.random.randint(6, 60, size=n)

    default = (
        (income < 1500).astype(int)
        + (credit_score < 600).astype(int)
        + (employment_type == "Unemployed").astype(int)
    )
    default = (default > 1).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income.astype(int),
        "loan_amount": loan_amount.astype(int),
        "employment_type": employment_type,
        "credit_score": credit_score.astype(int),
        "tenure": tenure,
        "default": default
    })
    return df

df = get_data()
X = df.drop(columns="default")
y = df["default"]

# Predict scores
probs = model.predict_proba(X)[:, 1]

def prob_to_score(p, base_score=600, base_odds=50, pdo=20):
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)

scores = prob_to_score(probs)

df["predicted_prob"] = probs
df["model_score"] = scores.astype(int)

# Display data
st.subheader("Sample Predictions")
st.dataframe(df.head(10))

# Select a row
idx = st.slider("Select row index for SHAP explanation", 0, len(df)-1, 0)

# SHAP Explanation
X_transformed = preprocessor.transform(X)
explainer = shap.Explainer(model.named_steps["clf"], X_transformed)
shap_values = explainer(X_transformed)

st.subheader(f"SHAP Explanation for Row {idx}")
shap.plots.waterfall(shap_values[idx], show=False)
fig = plt.gcf()  # get current figure
st.pyplot(fig)
