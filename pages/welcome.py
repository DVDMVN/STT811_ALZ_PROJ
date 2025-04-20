import streamlit as st

st.title("Welcome to the Alzheimer's Insights Dashboard 🧠")
st.header("Exploring risk factors for Alzheimer's disease through data", divider=True)

st.subheader("About this Project")
st.write(
    """
    This dashboard presents an interactive exploration of a global Alzheimer's prediction dataset sourced from Kaggle.
    Our goal is to uncover key risk factors for Alzheimer's diagnosis using data-driven analysis and machine learning techniques.
    """
)

st.write(
    """
    Contents: This application is based on a single, rich dataset containing demographic, lifestyle, and genetic information from individuals across various countries.
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