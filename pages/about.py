import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import io
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import time
from functools import wraps
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_validate

def time_it():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[TIMER] {func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper
    return decorator

@time_it()
def evaluate_model(estimator, X, y, cv):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    cv_results = cross_validate(estimator, X, y, scoring=scoring, cv=cv, n_jobs=-1)

    metrics = {
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision_macro"]),
        "recall": np.mean(cv_results["test_recall_macro"]),
        "f1": np.mean(cv_results["test_f1_macro"]),
    }
    return metrics

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
    st.pyplot(fig)


def main():
    st.title("About the Data 💾")

    DATA_URL_1 = ('data/alzheimers.csv')
    DATA_URL_2 = ('data/alzheimers_encoded.csv')
    alzheimers = pd.read_csv(DATA_URL_1)
    alzheimers_encoded = pd.read_csv(DATA_URL_2)

    X = alzheimers_encoded.drop(columns=["Alzheimers_Diagnosis_Yes"])
    y = alzheimers_encoded["Alzheimers_Diagnosis_Yes"]
    feature_names = X.columns

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
    results = {name: evaluate_model(model, X, y, skf) for name, model in estimators.items()}

    st.write(
        """ 
            Our dataset is a collection of patient observations documenting risk factors associated with Alzheimer's disease, as well as their specific diagnosis. 
            The data has a global perspective, coming from 20 different countries across the world, including the United States, United Kingdom, China, India, Brazil and many more, 
            with an even spread of roughly 3700 records per country. 
            The data was gathered and made public by a user on Kaggle. 
            Many of the features in our dataset are frequently cited in popular scientific literature as potential links to development and progression of various dementia symptoms and Alzheimer's disease variants, 
            making it prime for investigating relevant claims about Alzheimer's risk factors.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Original Data Set", "Preprocessing", "Analysis", "Feature Selection"])

    with tab1:
        st.subheader("Original Data Set")
      
        st.markdown("Dataset source: [Kaggle](https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global)")

        st.write("Sample Data:")
        st.dataframe(alzheimers)

        st.write(
            """
              The data is tabular, with a mixture of 24 different qualitative and quantitative features.
            """
        )

    with tab2:
        st.subheader("Preprocessing")
      
        st.markdown(
          """
              **Preprocessing Steps:**
              
              1. Standardization of numerical features  
                 - All continuous variables will be standardized (zero mean, unit variance).
              
              2. Feature-appropriate encoding methods
                 - **One-hot encoding** for **nominal variables** (no natural order).  
                 - **Ordinal encoding** for **ordered categorical variables** (with natural order).  
                 - **Label encoding** for **binary categorical features** (e.g., Yes/No for the target).
              
              3. Column name normalization
                 - Convert column names to lowercase, replace spaces with underscores, and removed apostraphes.
          """
        )

        st.write(
            """
                For specific operations, please refer to the file preprocessing.py from the source library.
            """
        )

        st.write(
            """
                **New data set after processing:**
            """
        )

        st.dataframe(alzheimers_encoded)
    
    with tab3:
        st.subheader("Analysis")

        st.write("**Summary Statistics:**")
        st.write("We first want to understand the structure of the dataset, including the size of the dataset, qualitative and quantitative features.")

        buffer = io.StringIO()
        alzheimers.info(buf=buffer)
        st.text(buffer.getvalue())

        st.dataframe(alzheimers.describe())

        st.write(
            """
                Quantitative (4): Age, BMI, Cognitive Test Score, Education Level
            """
        )

        st.write(
            """
                Qualitative (20): Physical Activity Level, Alcohol Consumption, Stress Levels, Country, Diabetes, Smoking Status, ...
            """
        )

        st.write(
            """
                This dataset appears to be very clean. 
                No missing values, the row counts for each attribute remain consistent for all. 
                Data types appear as expected.
                Frequency counts for categorical variables show a good distribution for each.
            """
        )
      
        st.write("**Linear correlation analysis:**")
      
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(alzheimers.select_dtypes(include='number').corr(), ax=ax, vmin=-1, vmax=1, cmap='coolwarm', annot=True, square=True)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        st.pyplot(fig)

        st.write(
            """
              The heatmap shows that most features have no or only limited direct linear correlation with diagnosis.
            """
        )

        st.write(
            """
              We would like to refer to categorical features to explore more influencing factors of Alzheimer's disease. 
              We will find the most valuable features and perform feature selection in subsequent analysis.
            """
        )
      
    with tab4:
        st.subheader("Feature Selection")

        st.write("Exploratory analysis based on the coded dataset.")
      
        st.write(
            """
              Only some models have feature_importances_ or coef_, the most straightforward metrics for feature importance. 
              We will only be doing feature importance analysis on those models that have such attributes:
            """
        )

        run_feature_importance_analysis(estimators, X, y, feature_names)

        st.write(
            """
              Find the top 5 most important features through visualization：
            """
        )
      
        plot_feature_importance(estimators["RandomForest"], X, y, feature_names, "RandomForest")
        
if __name__ == "__main__":
    main()

