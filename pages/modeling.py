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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE

from collections import defaultdict

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
    "SVM": LinearSVC(C=1.0),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

# -------------- FUNCTIONS --------------


@st.cache_data()
def evaluate_model(estimator_name, X, y):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    cv_results = cross_validate(
        estimators[estimator_name], X, y, scoring=scoring, cv=skf, n_jobs=-1
    )

    metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision_macro"]),
        "recall": np.mean(cv_results["test_recall_macro"]),
        "f1": np.mean(cv_results["test_f1_macro"]),
    }
    return metrics


def get_feature_importances(model):
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


# @st.cache_data()
# def run_feature_importance_analysis(X, y, num_importances=5):
#     fitted_models = {}
#     for name, model in estimators.items():
#         print(f"Fitting {name} ...")
#         model.fit(X, y)
#         fitted_models[name] = model

#     for name, model in fitted_models.items():
#         importances = get_feature_importances(model)
#         if importances is not None:
#             print(f"\n{name} feature importances:")
#             for feat, val in sorted(importances, key=lambda x: x[1], reverse=True)[
#                 :num_importances
#             ]:
#                 print(f"\t{feat}: {val:.4f}")
#         else:
#             print(f"\n{name} does not provide a direct feature importance measure.")


# @st.cache_data()
# def plot_feature_importance(model_name, X, y, num_importances=5):
#     if model_name not in estimators:
#         st.warning(f"Model '{model_name}' not found.")
#         return

#     st.write(f"Training model: {model_name}")
#     model = estimators[model_name]
#     model.fit(X, y)
#     st.success("Model training complete.")

#     importances = get_feature_importances(model)
#     if importances is None:
#         st.warning(f"{model_name} does not provide feature importance.")
#         return

#     sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
#     top_importances = sorted_importances[:num_importances]

#     labels = [t[0] for t in top_importances]
#     values = [t[1] for t in top_importances]

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.barh(labels[::-1], values[::-1])  # Plot from highest to lowest
#     ax.set_title(f"Top {num_importances} Feature Importances: {model_name}")
#     ax.set_xlabel("Importance")
#     ax.set_ylabel("Feature")
#     return fig


@st.cache_data()
def plot_feature_importance(model_name, X, y, num_importances=5):
    if model_name not in estimators:
        st.warning(f"Model '{model_name}' not found.")
        return

    model = estimators[model_name]
    model.fit(X, y)
    st.success(f"✅ Model '{model_name}' finished fitting!")

    importances = get_feature_importances(model)
    if importances is None:
        st.warning(f"{model_name} does not provide feature importance.")
        return

    sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
    top_importances = sorted_importances[:num_importances]

    labels = [t[0] for t in top_importances]
    values = [t[1] for t in top_importances]

    fig = px.bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title=f"Top {num_importances} Feature Importances: {model_name}",
    )
    # fig.update_layout(
    #     yaxis=dict(tickfont=dict(size=12)),
    #     xaxis=dict(tickfont=dict(size=12)),
    #     margin=dict(l=100, r=20, t=50, b=50)
    # )
    return fig


# -------------- VARIABLES --------------

alzheimers, alzheimers_encoded = load_data()
X = alzheimers_encoded.drop(columns=["Alzheimers_Diagnosis_Yes"])
y = alzheimers_encoded["Alzheimers_Diagnosis_Yes"]
feature_names_orig = alzheimers.columns
feature_names = X.columns

# -------------- PAGE --------------

st.title("Modeling 📊")

st.divider()

st.write(
    """
    TODO:
    Our goal is two-fold, 
    - Finding precise models for true Alzheimer's detection
    - Understanding underlying importances used by the models
    """
)

st.header("Benchmark Model", divider=True)

benchmark_age = 40

st.write(
    """
    As a trivial benchmark to compare against, we postulate a model that simply chooses patients above age {benchmark_age} to have Alzheimers.
    """
)

st.write(
    """
    SHOW BENCHMARK MODEL
    """
)

st.write(
    """
    Such a model already gives an accuracy of {}!
    - This benchmark serves to...
    """
)

st.header("Maching Learning Analysis", divider=True)

st.write(
    """
    For our machine learning analysis we utilized 8 different classification models
    - LogisticRegression
    - RandomForest
    - XGBoost
    - NaiveBayes
    - LDA
    - QDA
    - KNN
    - SVM


    Our target for classification is  feature “Alzheimer’s Diagnosis”, a binary categorical feature indicating whether a participant has been diagnosed with Alzheimer’s.
    - 0: negative diagnosis for Alzheimer’s
    - 1: positive diagnosis for Alzheimer’s

    We used our set of full encoded and standardized features for our predictors:
    - TODO
    - TODO
    """
)

st.subheader("Training Models", divider=True)
# --- Progress bar ---
progress_bar = st.progress(0, text="Evaluating models...")
results = {}

# --- Model evaluation loop ---
for i, (name, _) in enumerate(estimators.items(), start=1):
    with st.spinner(f"Training and testing {name}..."):
        results[name] = evaluate_model(name, X, y)
        progress_bar.progress(
            i / len(estimators), text=f"Completed {i}/{len(estimators)} models"
        )

# --- Done! ---
st.toast("✅ All models have been evaluated!", icon="🎉")
st.success("✅ Model evaluation complete.")

st.subheader("Model Metrics", divider=True)

metric_totals = defaultdict(float)

for metrics in results.values():
    for metric_name, value in metrics.items():
        if metric_totals[metric_name]:
            metric_totals[metric_name] += value
        else:
            metric_totals[metric_name] = value

average_metrics = {
    metric_name: total / len(results) for metric_name, total in metric_totals.items()
}

ranking_metric = "f1"  # <- change to "accuracy", "precision", etc. if desired
sorted_models = sorted(
    results.items(), key=lambda item: item[1][ranking_metric], reverse=True
)

medal_emojis = ["🥇", "🥈", "🥉"]
medals = {
    name: medal
    for medal, (name, _) in zip(medal_emojis, sorted_models)  # only first 3 get a medal
}

for name, metrics_dict in results.items():
    label = f"{medals.get(name, '')} {name}".strip()

    model_col, accuracy_col, precision_col, recall_col, f1_col = st.columns(5)
    model_col.write(label)
    accuracy_col.metric(
        "Accuracy",
        f"{metrics_dict['accuracy'] * 100:.2f}%",
        f"{(metrics_dict['accuracy'] - average_metrics['accuracy']) * 100:.2f}%",
    )
    precision_col.metric(
        "Precision",
        f"{metrics_dict['precision'] * 100:.2f}%",
        f"{(metrics_dict['precision'] - average_metrics['precision']) * 100:.2f}%",
    )
    recall_col.metric(
        "Recall",
        f"{metrics_dict['recall'] * 100:.2f}%",
        f"{(metrics_dict['recall'] - average_metrics['recall']) * 100:.2f}%",
    )
    f1_col.metric(
        "F1-Score",
        f"{metrics_dict['f1'] * 100:.2f}%",
        f"{(metrics_dict['f1'] - average_metrics['f1']) * 100:.2f}%",
    )

st.write(
    """
    TODO:
    """
)

st.subheader("Feature Importances", divider=True)


st.write("Exploratory analysis based on the coded dataset.")

st.write(
    """
    Only some models have feature_importances_ or coef_, the most straightforward metrics for feature importance. 
    We will only be doing feature importance analysis on those models that have such attributes:
    """
)

st.write(
    """
        Find the most important features through visualization：
    """
)

logreg_tab, randf_tab, xgb_tab, lda_tab, svm_tab = st.tabs(
    ["LogisticRegression", "RandomForest", "XGBoost", "LDA", "SVM"]
)

with logreg_tab:
    logreg_num_importances = st.slider(
        label="Number of importances:", min_value=5, max_value=20, value=5, key="logreg"
    )
    with st.spinner(f"Training model: {'Logistic Regression'}"):
        st.plotly_chart(
            plot_feature_importance("LogisticRegression", X, y, logreg_num_importances)
        )
with randf_tab:
    randf_num_importances = st.slider(
        label="Number of importances:", min_value=5, max_value=20, value=5, key="randf"
    )
    with st.spinner(f"Fitting model: {'Random Forest'}"):
        st.plotly_chart(
            plot_feature_importance("RandomForest", X, y, randf_num_importances)
        )
with xgb_tab:
    xgb_num_importances = st.slider(
        label="Number of importances:", min_value=5, max_value=20, value=5, key="xgb"
    )
    with st.spinner(f"Fitting model: {'XGBoost'}"):
        st.plotly_chart(plot_feature_importance("XGBoost", X, y, xgb_num_importances))
with lda_tab:
    lda_num_importances = st.slider(
        label="Number of importances:", min_value=5, max_value=20, value=5, key="lda"
    )
    with st.spinner(f"Fitting model: {'LDA'}"):
        st.plotly_chart(plot_feature_importance("LDA", X, y, lda_num_importances))
with svm_tab:
    svm_num_importances = st.slider(
        label="Number of importances:", min_value=5, max_value=20, value=5, key="svm"
    )
    with st.spinner(f"Fitting model: {'SVM'}"):
        st.plotly_chart(plot_feature_importance("SVM", X, y, svm_num_importances))

st.write(
    """
    TODO:
    """
)

st.header("Interaction Feature Analysis", divider=True)

st.write(
    """
    TODO:
    """
)

st.subheader("Correlation Analysis")

st.write(
    """
    TODO:
    """
)

st.subheader("Modeling with Interaction Features")

st.write(
    """
    Due to limitations with memory, modeling with such an extreme number of features is too expensive for most models. For this reason, we will only be modeling
    with interaction features as predictors for a select few models that run in a reasonable amount of time.
    """
)

st.write(
    """
    """
)

st.subheader("Importances of Original Features vs Interaction Features")

st.write(
    """
    """
)

st.header("Conclusions", divider=True)

st.write(
    """
    HERE SOME CONCLUSIONS
    """
)
