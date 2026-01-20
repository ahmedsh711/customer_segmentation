import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.mixture import GaussianMixture
from .utils import aggregate_customer_features

# Settings
DATA_PATH = 'data/raw/Cleaned_Data_Merchant_Level_2.csv'
MODEL_PATH = 'models/best_model.pkl'
COLS_PATH = 'models/feature_columns.pkl'
PARQUET_PATH = 'data/preprocessed/preprocessed_df.parquet'

def train():
    print("Loading Data..")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    print("Processing Data...")
    customer_features = aggregate_customer_features(df)
    
    # Columns to use for training
    training_cols = [
        'Recency', 'Frequency', 'Monetary_Total', 
        'Monetary_Max', 'Category_Diversity', 
        'Total_Points', 'Customer_Tenure'
    ]
    
    X = customer_features[training_cols].fillna(0)
    
    print("Training Model...")
    # Pipeline: Log -> Scale -> Cluster
    model = Pipeline(steps=[
        ('Log', FunctionTransformer(np.log1p, validate=False)),
        ('Scaler', RobustScaler()),
        ('GMM', GaussianMixture(n_components=5, covariance_type='tied', random_state=42))
    ])
    
    labels = model.fit_predict(X)
    
    print("Saving Files..")
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../data/preprocessed', exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(training_cols, COLS_PATH)
    
    # Save data for the dashboard
    customer_features['Cluster'] = labels
    customer_features.to_parquet(PARQUET_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    train()