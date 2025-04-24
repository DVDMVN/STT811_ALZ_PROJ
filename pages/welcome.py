import streamlit as st
import plotly.graph_objects as go

st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apparently streamlit doesn't have centering for their elements still 😝. CSS hack
st.markdown('<h1 class="title">Welcome to the Alzheimer\'s Insights Dashboard 🧠</h1>', unsafe_allow_html=True)

st.header("Exploring risk factors for Alzheimer's disease through tabular data", divider=True)

st.write(
    """
    #### Motivation
    
    Alzheimer's disease is an irreversible, progression neurodegenerative disorder characterized by memory loss, mental dysfunction, 
    and changes in personality. Eventually, neuronal damage of Alzheimer's becomes so extensive, parts of the brain that enable basic bodily functions, 
    such as walking or swallowing, may be limited. Loss of mobility paired with cognitive limitations causes required on the clock care. Utimately, the disease is fatal.
    
    Based on the AA (Alzheimer's Association) 2024 annual review [Link](https://www.alz.org/getmedia/76e51bb6-c003-4d84-8019-e0779d8c4e8d/alzheimers-facts-and-figures.pdf):
    - Alzheimer's is the most common cause of dementia, accounting for an estimated 60 - 80% of cases.
    - About 1 in 9 people (10.9%) age 65 and older has Alzheimer's dementia.
    """
)

@st.cache_data()
def plot_AA_agegroup_alz_percentages():
    # Based on data from 'prevalence' section of this review by the AA: https://www.alz.org/getmedia/76e51bb6-c003-4d84-8019-e0779d8c4e8d/alzheimers-facts-and-figures.pdf
    age_groups = ['65 - 74', '75 - 84', '85+']
    percentages = [5.0, 13.2, 33.4]

    fig = go.Figure(
        data=[
            go.Bar(
                x=age_groups,
                y=percentages,
                text=[f"{p}%" for p in percentages],
            )
        ]
    )

    fig.update_layout(
        title="[US] Percentage of Alzheimer's prevalence in older age groups",
        xaxis_title="age group",
        yaxis_title="percentage",
        yaxis_range=[0, 40],
    )
    return fig

st.plotly_chart(plot_AA_agegroup_alz_percentages())

st.write(
    """
    Though Alzheimer's and other dimentia related diseases are among the most researched in the world, the causes of Alzheimer's disease remain poorly understood. [Link](https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet) Still, many
    prevaling risk factors emerge from popular literature and scientific articles alike. Among the most commonly referenced are, age, genetics, and health / lifestyle factors.
    [Link](https://www.nia.nih.gov/health/alzheimers-causes-and-risk-factors/what-causes-alzheimers-disease)
    [Link](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447)
    """
)

st.write(
    """    
    #### Problem Statement
    
    Our research has two primary objectives:
    - To evaluate the effectiveness of machine learning models in accurately diagnosing Alzheimer's disease based on tabular data (non-image data), including demographic, 
    medical, and lifestyle factors.
    - To identify and quantify the most significant risk factors to inform intervention, prevention, and diagnostic strategies.
    """
)

st.write(
    """
    #### Contents
    This application is based on a single, rich dataset containing demographic, lifestyle, and genetic information from individuals across 
    various countries. Source: [kaggle](https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global)
    - The dashboard is organized across three pages:
        - **About the Data**
            - Initial Data Analysis (IDA) and Exploratory Data Analysis (EDA)
            - Handling of missing values and outliers
            - Feature encoding and other preprocessing techniques
            - Visualization and analysis of feature distributions and correlations
        - **Modeling**: Evaluation of different machine learning models, including feature importance and interaction analysis.
            - Performance of multiple ML models on Alzheimer's diagnosis prediction
            - Analysis of top contributing features
            - Exploration of interaction effects and their impact on model performance
        - **Documentation**: Technical appendix covering feature details and references for sources.
    """
)

st.markdown(
    """
    ---
    ✅ Ready to begin? Use the sidebar to navigate through the dashboard.
    """
)

# TODO: Insert teaser animations or key metric previews here

st.write("🔗 For more information or to view the full documentation, please visit our [GitHub Repository](https://github.com/DVDMVN/STT811_ALZ_PROJ).")

st.balloons()