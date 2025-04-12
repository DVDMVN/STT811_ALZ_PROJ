import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

"""
- Standardization for numerical variables
- Ordinal encoding for ordinal categorical variables
- OneHot encoding for nominal categorical variables
- Also normalized the naming scheme for variables: whitespace to _, and removed apostraphes.
"""
def alzheimers_preprocessing(alzheimers: DataFrame) -> DataFrame:
    numerical_columns = alzheimers.select_dtypes(include = "number").columns

    ordinal_columns = {
        'Physical Activity Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Alcohol Consumption': {'Never': 0, 'Occasionally': 1, 'Regularly': 2},
        'Cholesterol Level': {'Normal': 0, 'High': 1},
        'Depression Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Sleep Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
        'Dietary Habits': {'Unhealthy': 0, 'Average': 1, 'Healthy': 2},
        'Air Pollution Exposure': {'Low': 0, 'Medium': 1, 'High': 2},
        'Social Engagement Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Income Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Stress Levels': {'Low': 0, 'Medium': 1, 'High': 2},
    }

    non_binary_nominal_columns = [
        'Country',
        'Smoking Status',
        'Employment Status',
        'Marital Status',
    ]

    binary_nominal_columns = [
        'Gender',
        'Diabetes',
        'Hypertension',
        'Family History of Alzheimer’s',
        'Genetic Risk Factor (APOE-ε4 allele)',
        'Urban vs Rural Living',
        'Alzheimer’s Diagnosis'
    ]

    standard_scaler_transformer = StandardScaler()

    def get_ordinal_categories(mapping):
        return [list(mapping[col].keys()) for col in mapping]

    ordinal_column_names = list(ordinal_columns.keys())
    ordinal_categories = get_ordinal_categories(ordinal_columns)

    ordinal_transformer = OrdinalEncoder(categories=ordinal_categories)

    onehot_nonbinary_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    onehot_binary_transformer = OneHotEncoder(drop='if_binary', handle_unknown='ignore')

    preprocessor = ColumnTransformer( # Using ColumnTransformer to just conglomerate the different transformers used, not nessecary
        transformers=[
            ("numerical", standard_scaler_transformer, numerical_columns),
            ("ordinal", ordinal_transformer, ordinal_column_names),
            ("onehot_nonbinary", onehot_nonbinary_transformer, non_binary_nominal_columns),
            ("onehot_binary", onehot_binary_transformer, binary_nominal_columns),
        ],
        remainder="passthrough"
    )

    alzheimers_encoded = preprocessor.fit_transform(alzheimers)

    # --- After transformation, we lose our feature names, we can add them back in:

    # non-binary nominal feature names
    onehot_nonbinary_encoder = preprocessor.named_transformers_["onehot_nonbinary"]
    nonbinary_feature_names = onehot_nonbinary_encoder.get_feature_names_out(non_binary_nominal_columns)

    # binary nominal feature names
    onehot_binary_encoder = preprocessor.named_transformers_["onehot_binary"]
    binary_feature_names = onehot_binary_encoder.get_feature_names_out(binary_nominal_columns)

    all_feature_names = (
        list(numerical_columns)
        + list(ordinal_column_names)
        + list(nonbinary_feature_names)
        + list(binary_feature_names)
    )
    alzheimers_encoded = pd.DataFrame(alzheimers_encoded, columns=all_feature_names)

    # Replace whitespace with underscore and remove weird apostraphe
    alzheimers_encoded.columns = (
        alzheimers_encoded.columns
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("’", "", regex=False)
    )
    return alzheimers_encoded