
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 2000

# Generate features
age = np.random.randint(18, 65, size=n)
income = np.random.normal(2500, 1000, size=n).clip(500, None)
loan_amount = np.random.normal(10000, 5000, size=n).clip(1000, None)
employment_type = np.random.choice(["Salaried", "Self-Employed", "Unemployed"], size=n, p=[0.6, 0.3, 0.1])
credit_score = np.random.normal(650, 50, size=n).clip(300, 850)
tenure = np.random.randint(6, 60, size=n)

# Generate target variable with logic
# More likely to default if income is low, credit_score is low, or unemployed
default = (
    (income < 1500).astype(int)
    + (credit_score < 600).astype(int)
    + (employment_type == "Unemployed").astype(int)
)
default = (default > 1).astype(int)  # mark as defaulter if at least 2 high-risk conditions

# Assemble DataFrame
df = pd.DataFrame({
    "age": age,
    "income": income.astype(int),
    "loan_amount": loan_amount.astype(int),
    "employment_type": employment_type,
    "credit_score": credit_score.astype(int),
    "tenure": tenure,
    "default": default
})


# Assume df is your synthetic dataset
X = df.drop(columns="default")
y = df["default"]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing
num_cols = ["age", "income", "loan_amount", "credit_score", "tenure"]
cat_cols = ["employment_type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

# Model pipeline
model_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

# Fit the model
model_pipeline.fit(X_train, y_train)

def prob_to_score(p, base_score=600, base_odds=50, pdo=20):
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds)

# Predict probability of default
probs = model_pipeline.predict_proba(X_val)[:, 1]
scores = prob_to_score(probs)

# Add to DataFrame
results_df = X_val.copy()
results_df["default_prob"] = probs
results_df["credit_score"] = scores.astype(int)
results_df["true_default"] = y_val.values

import shap

# Transform data for SHAP
X_val_transformed = model_pipeline.named_steps["pre"].transform(X_val)

# Initialize SHAP explainer on the logistic model
explainer = shap.Explainer(model_pipeline.named_steps["clf"], X_val_transformed)

# Compute SHAP values
shap_values = explainer(X_val_transformed)

import joblib
joblib.dump(model_pipeline, "loan_default_model.pkl")
joblib.dump(preprocessor, "loan_preprocessor.pkl")