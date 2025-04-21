import streamlit as st

st.title("Documentation: 📔")

st.divider()

st.subheader("Features details:", divider = True)

# Numerical Columns
st.write("**Numerical Features**")

st.markdown("""
- `Age` - Participant's age in years.  
- `Education Level` - Number of years or level of formal education completed.  
- `BMI` - Body Mass Index, a measure of body fat based on height and weight.  
- `Cognitive Test Score` - Score from a standardized cognitive assessment (higher = better performance).
""")

# Ordinal Columns
st.write("**Ordinal Features**")

st.markdown("""
- `Physical Activity Level` - Frequency or intensity of physical activity (Low, Medium, High).  
- `Alcohol Consumption` - Frequency of alcohol intake (Never, Occasionally, Regularly).  
- `Cholesterol Level` - Classification of cholesterol (Normal, High).  
- `Depression Level` - Severity of depressive symptoms (Low, Medium, High).  
- `Sleep Quality` - Self-reported sleep quality (Poor, Average, Good).  
- `Dietary Habits` - Overall diet healthiness (Unhealthy, Average, Healthy).  
- `Air Pollution Exposure` - Estimated exposure to air pollution (Low, Medium, High).  
- `Social Engagement Level` - Frequency of social interactions and involvement (Low, Medium, High).  
- `Income Level` - Income category or bracket (Low, Medium, High).  
- `Stress Levels` - Self-perceived stress level (Low, Medium, High).
""")

# Binary Nominal Columns
st.write("**Binary Nominal Features**")

st.markdown("""
- `Gender` - Gender identity (Male, Female).  
- `Diabetes` - Whether the participant has been diagnosed with diabetes (Yes, No).  
- `Hypertension` - Whether the participant has high blood pressure (Yes, No).  
- `Family History of Alzheimer’s` - Whether there's a family history of Alzheimer’s (Yes, No).  
- `Genetic Risk Factor (APOE-ε4 allele)` - Presence of the APOE-ε4 genetic risk allele (Yes, No).  
- `Urban vs Rural Living` - Type of residence location (Urban, Rural).  
- `Alzheimer’s Diagnosis` ⭐Target⭐ - Whether participant has been diagnosed with Alzheimer’s (Yes, No).
""")

# Non-binary Nominal Columns
st.write("**Non-binary Nominal Features**")

st.markdown("""
- `Country` - Country of residence.  
- `Smoking Status` - Smoking history (Never, Former, Current).  
- `Employment Status` - Current employment state (Employed, Retired, Unemployed).  
- `Marital Status` - Marital condition (Single, Married, Widowed).
""")

st.subheader("Sources:", divider = True)

st.markdown(
    """
    - [[1]](https://www.alz.org/alzheimers-dementia/what-is-alzheimers/causes-and-risk-factors) https://www.alz.org/alzheimers-dementia/what-is-alzheimers/causes-and-risk-factors
    - [[2]](https://www.alzint.org/about/risk-factors-risk-reduction/) https://www.alzint.org/about/risk-factors-risk-reduction/
    - [[3]](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447) https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447
    """
)