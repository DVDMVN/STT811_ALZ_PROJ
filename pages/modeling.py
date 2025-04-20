import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_validate
from util import load_data


estimators = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
    "NaiveBayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "KNN": KNeighborsClassifier(),
    "SVM": LinearSVC(C=1.0)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

@st.cache_data()
def evaluate_model(estimator_name, X, y):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    cv_results = cross_validate(estimators[estimator_name], X, y, scoring=scoring, cv=skf, n_jobs=-1)

    metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision_macro"]),
        "recall": np.mean(cv_results["test_recall_macro"]),
        "f1": np.mean(cv_results["test_f1_macro"]),
    }
    return metrics

@st.cache_data()
def get_feature_importances(model, feature_names):
    # Tree based models usually have a feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return list(zip(feature_names, importances))

    # Linear models usually have a coef_ attribute
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0])

        return list(zip(feature_names, importances))
    else:
        return None

@st.cache_data()
def run_feature_importance_analysis(estimators, X, y, feature_names, num_importances=5):
    fitted_models = {}
    for name, model in estimators.items():
        print(f"Fitting {name} ...")
        model.fit(X, y)
        fitted_models[name] = model

    for name, model in fitted_models.items():
        importances = get_feature_importances(model, feature_names)
        if importances is not None:
            print(f"\n{name} feature importances:")
            for feat, val in sorted(importances, key=lambda x: x[1], reverse=True)[:5]:
                print(f"\t{feat}: {val:.4f}")
        else:
            print(f"\n{name} does not provide a direct feature importance measure.")

@st.cache_data()
def plot_feature_importance(model_name, estimators, X, y, feature_names, num_importances=5):
    if model_name not in estimators:
        st.warning(f"Model '{model_name}' not found.")
        return

    st.write(f"Training model: {model_name}")
    model = estimators[model_name]
    model.fit(X, y)
    st.success("Model training complete.")

    importances = get_feature_importances(model, feature_names)
    if importances is None:
        st.warning(f"{model_name} does not provide feature importance.")
        return

    sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
    top_importances = sorted_importances[:num_importances]

    labels = [t[0] for t in top_importances]
    values = [t[1] for t in top_importances]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1])  # Plot from highest to lowest
    ax.set_title(f"Top {num_importances} Feature Importances: {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")


alzheimers, alzheimers_encoded = load_data()
X = alzheimers_encoded.drop(columns=["Alzheimers_Diagnosis_Yes"])
y = alzheimers_encoded["Alzheimers_Diagnosis_Yes"]
feature_names_orig = alzheimers.columns
feature_names = X.columns

# --- Progress bar ---
progress_bar = st.progress(0, text="Evaluating models...")
results = {}

# --- Model evaluation loop ---
for i, (name, _) in enumerate(estimators.items(), start=1):
    with st.spinner(f"Training {name}..."):
        results[name] = evaluate_model(name, X, y)
        progress_bar.progress(i / len(estimators), text=f"Completed {i}/{len(estimators)} models")

# --- Done! ---
st.toast("✅ All models have been evaluated!", icon="🎉")
st.success("Model evaluation complete.")