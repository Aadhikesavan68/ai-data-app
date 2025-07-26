import pandas as pd
from sklearn.impute import KNNImputer
from scipy.stats import zscore

def clean_data(df):
    # Step 1: Extract numeric columns only
    df_numeric = df.select_dtypes(include='number')

    # Step 2: Check if we have numeric columns
    if df_numeric.empty:
        raise ValueError("❌ No numeric columns found in uploaded file. Please upload valid survey data.")

    # Step 3: Impute missing values
    imputer = KNNImputer(n_neighbors=2)
    imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Step 4: Remove outliers using Z-score method
    z_scores = abs(imputed.apply(zscore))
    cleaned = imputed[(z_scores < 3).all(axis=1)]

    return cleaned
