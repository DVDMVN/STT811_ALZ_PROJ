import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

import numpy as np

from imblearn.under_sampling import RandomUnderSampler

from util import load_data

# -------------- VARIABLES --------------

alzheimers, alzheimers_encoded = load_data()
alzheimers['Country'] = alzheimers['Country'].apply(lambda x: str(x))
X = alzheimers_encoded.drop(columns=["Alzheimers_Diagnosis_Yes"])
y = alzheimers_encoded["Alzheimers_Diagnosis_Yes"]
feature_names_orig = alzheimers.columns
feature_names = X.columns

# -------------- PAGE --------------

st.title("About the Data 💾")

st.divider()

st.write(
    """
    This page details the process for cleaning and preprocessing of the data before its usage in analysis and modeling. Each preprocessing decision 
    is made with ample reasoning and investigation, noting why changes were made may be useful in the assessment of our analysis. Additionally, this page will include our
    exploratory visual analysis that may justify further some explanation and decisions.
    """    
)

overview, analysis, preprocessing= st.tabs(["Dataset Overview", "Analysis", "Preprocessing"])

with overview:
    st.header("Dataset Overview", divider = True)

    st.write(
        """
        In this project, we utilize a dataset from kaggle: [Link to Dataset](https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global)

        This dataset is a collection of patient observations documenting risk factors associated with Alzheimer's disease, as well as their specific diagnosis.
        - The data has a global perspective, coming from 20 different countries across the world, including the United States, United Kingdom, China, India, Brazil and many more, 
        with an even spread of roughly 3700 records per country.
        - Many of the features in our dataset are frequently cited in popular scientific literature as potential links to development and progression of various dementia symptoms and Alzheimer's disease variants, 
        making it prime for investigating relevant claims about Alzheimer's risk factors.
        """
    )

    st.subheader("Dataset Preview:")
    st.dataframe(alzheimers.head(10))

    st.subheader("Basic Statistics and Shape")

    st.write(
        """
        The data is tabular, with a mixture of 24 different qualitative and quantitative features.
        - Quantitative (4): Age, BMI, Cognitive Test Score, Education Level
        - Qualitative (20): Physical Activity Level, Alcohol Consumption, Stress Levels, Country, Diabetes, Smoking Status, ...
        """
    )

    numerical_tab, categorical_tab = st.tabs(['Numerical Statistics', 'Categorical Statistics'])
    with numerical_tab:
        st.dataframe(alzheimers.describe())

    with categorical_tab:
        st.dataframe(alzheimers.select_dtypes(include="object").describe().astype(str))
    st.write(f"##### Shape: {alzheimers.shape}")

    st.write(
        """
        From these statistics, we can see that we have a very reasonable distribution / frequency count in both our categorical and numerical features.
        """
    )


    st.write("Learn more about each specific feature in our documentation:")
    st.page_link("pages/documentation.py", label="Documentation", icon="📔")

with analysis:
    st.header("Analysis", divider = True)

    st.subheader("Missingness and duplicate values analysis:")

    missing = (
        alzheimers.isna()
        .sum()
        .reset_index()
        .rename(columns={'index': 'feature', 0: 'num_missing'})
    )
    st.code( 
    '''
    missing = (
        alzheimers.isna()
        .sum()
        .reset_index()
        .rename(columns={'index': 'feature', 0: 'num_missing'})
    )
    st.write(missing)
    ''', language='python')
    st.write(missing)

    num_duplicates = alzheimers.duplicated().sum()
    st.code( 
    '''
    n_duplicates = alzheimers.duplicated().sum()
    st.write(f"Number of duplicate values = {num_duplicates}")
    ''', language='python')
    st.write(f"Number of duplicate values = {num_duplicates}")

    st.write(
        """
            This dataset appears to be very clean!
            - No missing values, the row counts for each attribute remain consistent for all.
            - No duplicate values, each row is unique.
        """
    )

    st.subheader("Bivariate distribution analysis")

    distribution_proportion_of_target_yes = alzheimers_encoded['Alzheimers_Diagnosis_Yes'].mean()
    distribution_proportion_of_target_no = 1 - distribution_proportion_of_target_yes

    st.write(
        f"""
        From our basic statistics, we can observe that the distribution for our target feature is slightly imbalanced. Without balancing:
        - Proportion of positive diagnosis = {distribution_proportion_of_target_yes * 100:.2f}%
        - Proportion of negative diagnosis = {distribution_proportion_of_target_no * 100:.2f}%
        
        To perform our bivariate distribution analysis (bivariate against target) we will perform two versions, one with our original imbalance, and another with data randomly undersampling the majority class:
        - The sample distribution diagnosis proportions are wildly different from common figures on the population distribution. Plotting both the original and the imbalance assists us in both investigating this
         discrepancy, as well as investigating a more truthful bivariance of our features against the target.
        """
    )

    @st.cache_data()
    def plot_bivariate_analysis(undersample: bool) -> plt.Figure:
        if undersample:
            rus = RandomUnderSampler(random_state = 1337)
            alzheimers_resampled, alzheimers_resampled_class = rus.fit_resample(
                alzheimers.drop("Alzheimer’s Diagnosis", axis=1),
                alzheimers["Alzheimer’s Diagnosis"],
            )
            alzheimers_resampled = pd.concat([alzheimers_resampled, alzheimers_resampled_class], axis = 1)
            data = alzheimers_resampled
        else:
            data = alzheimers

        fig, axes = plt.subplots(7, 4, figsize=(16, 24))
        axes: list[plt.Axes] = axes.ravel()

        def plot_kdeplot_bivariate(x, ax: plt.Axes):
            sns.kdeplot(
                data = data,
                x = x,
                hue = 'Alzheimer’s Diagnosis',
                ax = ax,
                legend = False
            )
            ax.set_title(x)
            ax.set_yticks([], [])

        def plot_histplot_bivariate(x, ax: plt.Axes):
            sns.histplot(
                data = data,
                x = x,
                hue = 'Alzheimer’s Diagnosis',
                ax = ax,
                multiple = 'dodge',
                legend = False
            )
            ax.set_title(x)
            ax.set_yticks([], [])

        # Numerical ones first
        numerical_columns = list(alzheimers.select_dtypes(include = 'number').columns)
        categorical_columns = list(alzheimers.select_dtypes(include = 'object').columns)

        ax_idx = 0
        for numerical_column in numerical_columns:
            plot_kdeplot_bivariate(numerical_column, axes[ax_idx])
            ax_idx += 1

        for categorical_column in categorical_columns:
            plot_histplot_bivariate(categorical_column, axes[ax_idx])
            ax_idx += 1
        axes[-1].remove()
        axes[-2].remove()
        axes[-3].remove()     

        plt.tight_layout()
        return fig
    
    bivariate_tab_orig, bivariate_tab_oversample = st.tabs(['Original', 'With undersampling'])
    with bivariate_tab_orig:
        st.write("**Using original Data (No undersampling)**")
        st.pyplot(plot_bivariate_analysis(undersample = False))

    with bivariate_tab_oversample:
        st.write("**Using data with random undersampling of majority class**")
        st.pyplot(plot_bivariate_analysis(undersample = True))

    st.write(
        """
        - Frequency counts for categorical features show a good distribution for each.
        """
    )

    st.subheader("Correlation analysis for numerical features:")
    
    st.write("Linear correlation heatmaps")

    @st.cache_data()
    def plot_correlation_heatmap(encoded = False):
        if encoded:
            corr_matrix = alzheimers_encoded.corr()
        else:
            alz_copy = alzheimers.copy()
            alz_copy["Alzheimer's Diagnosis"] = alzheimers['Alzheimer’s Diagnosis'].replace({
                'No': 0,
                'Yes': 1,
            })
            
            corr_matrix = alz_copy.select_dtypes(include = 'number').corr()
        corr_matrix = np.round(corr_matrix, 2)

        fig = px.imshow(
            corr_matrix,
            text_auto=True, 
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        
        fig.update_layout(
            title='Correlation heatmap',
            width=700,
            height=700,
            margin=dict(l=120, r=120, t=100, b=100)
        )
        
        return fig
    original_corr_tab, encoded_corr_tab = st.tabs(["Original", "After Encoding"])
    with original_corr_tab:
        st.plotly_chart(plot_correlation_heatmap(encoded = False))
    with encoded_corr_tab:
        st.plotly_chart(plot_correlation_heatmap(encoded = True))

    st.write(
        """
            We plot two correlation heatmaps, one on the original data and one with the encoded data after preprocessing.

            The heatmaps shows that all features seem to have little to no correlation with one another, and furthermore most do not seem to have any correlation with diagnosis.
            - The only notable correlations are with our target and the features age, family history, and genetic risk factor.
        """
    )

    st.write(
        """
        We would like to refer to categorical features to explore more influencing factors of Alzheimer's disease. 
        We find the most valuable features during machine learning through feature importances, contained within our modeling page.
        """
    )

    # @st.cache_data()
    # def plot_age_distribution_violin_plot():
    #     plot_data = alzheimers.copy()
    #     plot_data["Diagnosis_Label"] = plot_data["Alzheimer’s Diagnosis"].replace({
    #         'No': "negative diagnosis",
    #         'Yes': "positive diagnosis"
    #     })

    #     fig = px.violin(
    #         plot_data,
    #         x="Diagnosis_Label",
    #         y="Age",
    #         box=True,
    #         color="Diagnosis_Label",
    #         category_orders={"Diagnosis_Label": ["Negative Diagnosis", "Positive Diagnosis"]},
    #         title="Age Distribution by Alzheimer's Diagnosis"
    #     )

    #     fig.update_layout(
    #         xaxis_title="Alzheimer's Diagnosis",
    #         yaxis_title="Age",
    #         violingap=0.3,
    #         violingroupgap=0.1,
    #         violinmode='overlay'
    #     )
    #     return fig

    # st.plotly_chart(plot_age_distribution_violin_plot())


with preprocessing:
    st.subheader("Preprocessing")
    
    st.markdown(
        """
        Because our data appears very clean (no missing values or duplicates), we will not apply any imputation or removal of data.

        1. Standardization of numerical features  
            - All continuous features will be standardized (zero mean, unit variance).
        
        2. Feature-appropriate encoding methods
            - One-hot encoding for nominal features (no natural order).  
            - Ordinal encoding for ordered categorical features (with natural order).  
            - Label encoding for binary categorical features (e.g., Yes/No for the target).
        
        3. Column name normalization
            - Convert column names to lowercase.
            - Replace whitespace with underscores.
            - Remove apostraphes.
        """
    )

    st.write(
        """
        For specific operations, and our breakdown of feature types, please refer to the file preprocessing.py from the source library.
        """
    )

    st.write(
        """
        **Data set preview after preprocessing:** 
        """
    )

    st.dataframe(alzheimers_encoded.head(10))
    st.write(f"##### Shape: {alzheimers_encoded.shape}")
