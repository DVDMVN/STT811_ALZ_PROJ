import streamlit as st
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
        x=values,
        y=labels,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title=f"Top {num_importances} Feature Importances: {model_name}",
    )
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
    This page details the modeling approaches used to analyze Alzheimer's disease prediction factors. Provided are a demonstration of evaluation of multiple machine learning algorithms, feature / interaction feature importance analysis.
    """
)

st.write(
    """
    Reiterating our goals:
    - Evaluating the effectiveness of machine learning models in accurately diagnosing Alzheimer's disease based on tabular data (non-image data).
    - Identifying and quantify the most significant risk factors to inform intervention, prevention, and diagnostic strategies.
    """
)

st.header("Benchmark Model", divider=True)

benchmark_age = alzheimers['Age'].mean()

st.write(
    f"""
    As a trivial benchmark to compare against, we postulate a model that simply chooses patients above the mean age (mean = {benchmark_age:.2f}) to have Alzheimers.
    """
)

st.code(
    '''
    predictions = alzheimers['Age'] > alzheimers['Age'].mean()
    benchmark_accuracy = (predictions.values == alzheimers['Alzheimers_Diagnosis_Yes']).sum() / alzheimers.__len__()
    '''
, language='python')

benchmark_predictions = alzheimers_encoded['Age'] > 0
benchmark_accuracy = (benchmark_predictions.values == alzheimers_encoded['Alzheimers_Diagnosis_Yes']).sum() / alzheimers_encoded.__len__()

st.write(
    f"""
    Such a model already gives an accuracy of {benchmark_accuracy * 100:.2f}%!

    - Important note: True distribution of Alzheimer's for the population is not so uniformly distributed as our dataset, this accuracy does not truly represent
    population distribution.
        - For reference, the true metric is closer to about 1 in 9 people age 65 or older [[1]](https://www.alz.org/getmedia/76e51bb6-c003-4d84-8019-e0779d8c4e8d/alzheimers-facts-and-figures.pdf)
    """
)

st.write(
    """
    - The relatively good performance of this model highlights a strong association between age and Alzheimer's disease prevalence, which is very commonly referenced in research as the best predictor [[5]](https://www.alz.org/getmedia/dbc8fd3f-a1a8-4cfd-8c40-acb89fd65b23/annual-report-2024.pdf).
    - As this benchmark does not require any additional information other than testing for age, we use this result as a reference point to evaluate whether machine learning models 
    are able to incorporate richer feature sets and capture more complex patterns.
    """
)

st.header("Machine Learning Analysis", divider=True)

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
    - Though from our bivariate analysis in our about page,
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
    Among our eight models, Random Forest performed the best, followed by XGBoost, LDA, etc., with an accuracy of about 71%. 
    These models showed balanced precision, recall, and F1 scores, indicating strong and reliable performance.
    """
)

st.write(
    """
    In contrast, Naive Bayes, QDA, and KNN performed significantly worse (accuracy of about 63-66%). 
    This may be due to the fact that these models are not good at handling complex feature relationships or high-dimensional data.
    """
)

st.write(
    """
    In general, tree-based ensemble models and well-regularized linear classifiers are good choices for predicting Alzheimer's disease diagnosis. 
    In particular, Random Forest not only performs well in accuracy, but also performs better in other importance indicators. 
    Therefore, Random Forest is our ideal choice for predicting and analyzing risk factors.
    """
)

st.subheader("Feature Importances", divider=True)

st.write(
    """
    Note: Only some models have relevant direct feature importances:
    - "feature_importances_ or coef_, the most straightforward metrics for feature importance. 
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
    Key observations:
    - Age and genetic risk factor seem to be consistently the strongest signals, appearing at or near the top in every model that reports importances.
    - Family history of dementia seems to be the next-most influential variable.

    Some country of residence dummies climb the ranking in tree-based models, but their contributions are small. We suspect that this may be reflective of sample bias rather than genuine epidemiological differences, 
    though this area of Alzheimer's risk factor studies is one not very well researched [[1]](https://www.alz.org/getmedia/76e51bb6-c003-4d84-8019-e0779d8c4e8d/alzheimers-facts-and-figures.pdf).
    
    With this dataset, we are able to reproduce the most well known narrative regarding Alzheimer's risk association with age and other herditary factors, but we fail to reproduce supplementary, well tested, results regarding
      lifestyle variables such as smoking, obesity, and depression [[6]](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2272732#google_vignette).
    """
)

st.header("Interaction Feature Analysis", divider=True)

st.write(
    """
    In efforts to clarify why our results seem to only line up with the most popular narratives in age and hereditary factors, we additionally performed interaction feature analysis, hoping to uncover whether
    certain features, such as BMI or smoker status, may have significant predictive power in conjunction with one another. The interaction features we test are simple second degree cross multiples.
    """
)

st.caption(
    """
    (Due to limitations with memory and slow execution speed, modeling with such an extreme number of features is expensive. Our results have been prebaked and saved for the interaction feature analysis. Check the `master_notebook.ipynb` file
    for a full breakdown.)
    """
)

st.subheader("Interaction Correlation Analysis")

st.write(
    """
    A simple linear correlation analysis to qualify modeling which learns best from linear relation.
    """
)

st.caption("Values are Pearson coefficients - some NaN values come from the sparsity of the original encoded matrix causing divide by 0 errors in correlation calculation.")

interaction_correlation = pd.read_csv('cache/interaction_correlation.csv')
st.dataframe(interaction_correlation)

st.write(
    """
    We observe that the original features, without cross interaction, seem to retain highest correlation with the target feature.
    - While we hypothesized that feature interactions might reveal hidden patterns in Alzheimer's risk factors, our analysis shows that the primary individual features maintain their 
    dominance in linear correlation even after introducing interaction terms.
    - Specifically, Age, Genetic Risk Factor (APOE-ε4 allele), and Family History of Alzheimer's still emerge as top correlators, with their individual importance coefficients 
    significantly higher than any interaction terms.

    The absence of strong interaction effects suggests that these primary risk factors operate largely independently rather than synergistically in predicting Alzheimer's diagnosis.
    """
)

st.subheader("Modeling with Interaction Features")

interaction_feature_results_df = pd.read_csv('cache/model_metrics.csv')
interaction_feature_results = (
    interaction_feature_results_df.set_index("Model")
      .astype(np.float64)
      .to_dict(orient="index")
)

interaction_metric_totals = defaultdict(float)

for metrics in interaction_feature_results.values():
    for metric_name, value in metrics.items():
        if metric_totals[metric_name]:
            metric_totals[metric_name] += value
        else:
            metric_totals[metric_name] = value

average_interaction_metrics = {
    metric_name: total / len(results) for metric_name, total in metric_totals.items()
}

ranking_metric = "f1"  # <- change to "accuracy", "precision", etc. if desired
sorted_interaction_models = sorted(
    results.items(), key=lambda item: item[1][ranking_metric], reverse=True
)

medal_emojis = ["🥇", "🥈", "🥉"]
interaction_medals = {
    name: medal
    for medal, (name, _) in zip(medal_emojis, sorted_interaction_models)  # only first 3 get a medal
}

for name, metrics_dict in interaction_feature_results.items():
    label = f"{interaction_medals.get(name, '')} {name}".strip()

    model_col, accuracy_col, precision_col, recall_col, f1_col = st.columns(5)
    model_col.write(label)
    accuracy_col.metric(
        "Accuracy",
        f"{metrics_dict['accuracy'] * 100:.2f}%",
        f"{(metrics_dict['accuracy'] - results[name]['accuracy']) * 100:.2f}%",
    )
    precision_col.metric(
        "Precision",
        f"{metrics_dict['precision'] * 100:.2f}%",
        f"{(metrics_dict['precision'] - results[name]['precision']) * 100:.2f}%",
    )
    recall_col.metric(
        "Recall",
        f"{metrics_dict['recall'] * 100:.2f}%",
        f"{(metrics_dict['recall'] - results[name]['recall']) * 100:.2f}%",
    )
    f1_col.metric(
        "F1-Score",
        f"{metrics_dict['f1'] * 100:.2f}%",
        f"{(metrics_dict['f1'] - results[name]['f1']) * 100:.2f}%",
    )

st.write(
    """
    After rerunning modeling on our new set of interaction features, we see that most models, save for 'Naive Bayes' and the 'QDA' models, did not experience any significant movement.
    - This result is not suprising. From our correlation analysis we could already observe that independent signals (at least linearly) seem to be most powerful predictors.
    - The tree models seem to move the least. For tree models such as Random Forest and XGBoost, interactions between features are inherently captured by splitting and boosted residuals. 
    It seems that adding interaction features may be simply duplicating patterns they can learn natively.
    """
)

st.subheader("Importances of Original Features vs Interaction Features")

st.write(
    """
    To qualify our negligible difference in modeling metrics with interaction features, we examine the importances assigned to features in our models.
    """
)

lda_interaction_imp_df = pd.read_csv('cache/lda_interaction_importances.csv')
lr_interaction_imp_df = pd.read_csv('cache/logisticregression_interaction_importances.csv')
rf_interaction_imp_df = pd.read_csv('cache/randomforest_interaction_importances.csv')
svm_interaction_imp_df = pd.read_csv('cache/svm_interaction_importances.csv')
xgb_interaction_imp_df = pd.read_csv('cache/xgboost_interaction_importances.csv')

logreg_int_tab, randf_int_tab, xgb_int_tab, lda_int_tab, svm_int_tab = st.tabs(
    ["LogisticRegression", "RandomForest", "XGBoost", "LDA", "SVM"]
)

@st.cache_data()
def plot_interaction_feature_importances(df: pd.DataFrame, name: str):
    fig = px.bar(
        df.sort_values('Importance', ascending=False).head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top 10 feature importances (including interaction features): {name}'
    )
    return fig
with logreg_int_tab:
    st.plotly_chart(plot_interaction_feature_importances(lr_interaction_imp_df, "LogisticRegression"))
with randf_int_tab:
    st.plotly_chart(plot_interaction_feature_importances(rf_interaction_imp_df, "RandomForest"))
with xgb_int_tab:
    st.plotly_chart(plot_interaction_feature_importances(xgb_interaction_imp_df, "XGBoost"))
with lda_int_tab:
    st.plotly_chart(plot_interaction_feature_importances(lda_interaction_imp_df, "LDA"))
with svm_int_tab:
    st.plotly_chart(plot_interaction_feature_importances(svm_interaction_imp_df, "SVM"))

st.write(
    """
    Plotting the interaction feature trained model importances, we observe near identical importances listed before. 
    - Again, Age, Genetic Risk Factor (APOE-ε4 allele), and Family History of Alzheimer's seem to consistently emerge as top predictors.
    - Again, Random Forest seems to pickup slightly differing signals, also alloting significant importance to BMI, Cognitive Test Score, and Education Level.
    """
)

st.header("Conclusions", divider=True)

st.write(
    """
    Our analysis, both qualitatively and quantitatively with machine learning, points to tabular data on Alzheimer's disease being dominated in importance by a few strong predictors. These predictors also turn out to be
    largely out of control for individuals, namely:
    - Age
    - Genetic Risk Factor (APOE-ε4 allele)
    - Family History of Alzheimer's

    > This result lines up with the top three associated predictors found by the Alzheimer's Association [[1]](https://www.alz.org/getmedia/76e51bb6-c003-4d84-8019-e0779d8c4e8d/alzheimers-facts-and-figures.pdf).

    Ultimately, no particular modeling results were especially inspiring. Lifestyle and demographic data features from this dataset showed to very predictive poorly predictive, contrary to many articles [[6]](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2272732#google_vignette).

    With this in mind, when using tabular data we recommend keeping models simple.
    - Even a simple benchmark conditional only on age proves to be nearly as performant as our complex models.
    - Complex interaction models may not provide significant additional predictive power beyond simpler models focusing on age, genetic factors, and family history.

    From our machine learning analysis, while most models were similar in performance, we recommend non-linear tree ensembles as a primary model. Evidently, they are able to capture the interaction space without exhaustively adding these features.
    In this case, because of their indifference in performance after the addition of these terms, we regard these models as
     more robust than the others, able to capture the strongest signal amongst noisy or less relevant features.

    Modern literature points to usage of image data or a combination of image and tabular (multi-modal) for producing more precise and confident results [[7]](https://arxiv.org/abs/2305.19280) [[8]](https://arxiv.org/abs/2501.00861).
    For this reason, it is recommended to not rely on tabular data purely to conduct diagnosis.
    """
)
