import joblib
import pandas as pd
from pathlib import Path
import sys

# Define Segments
SEGMENTS = {
    0: 'Murky',
    1: 'At-Risk',
    2: 'Potential Loyalists',
    3: 'Loyal Customers',
    4: 'VIPs'
}

def load_resources():
    try:
        # 1. Get the folder where THIS file (inference.py) lives
        # This will be .../src/
        current_file_path = Path(__file__).resolve()
        src_dir = current_file_path.parent
        
        # 2. Go to the sibling 'models' folder
        # src/../models -> models/
        models_dir = src_dir.parent / "models"
        
        model_path = models_dir / "best_model.pkl"
        cols_path = models_dir / "feature_columns.pkl"

        # 3. Verify existence (Debug Check)
        if not model_path.exists():
            # This prints to the Streamlit Cloud logs
            print(f"ERROR: Model file missing! Looked at: {model_path}")
            # print directory contents to help debug
            if models_dir.exists():
                print(f"Contents of {models_dir}: {list(models_dir.iterdir())}")
            else:
                print(f"Models folder missing entirely at: {models_dir}")
            return None, None
            
        if not cols_path.exists():
            print(f"ERROR: Columns file missing at {cols_path}")
            return None, None

        # 4. Load
        model = joblib.load(model_path)
        cols = joblib.load(cols_path)
        
        return model, cols
        
    except Exception as e:
        print(f"Unexpected error loading resources: {e}")
        return None, None

def get_prediction_manual(data_dict):
    model, feature_cols = load_resources()
    
    if model is None:
        return "Error: Model file not found on server."

    # Convert single dictionary to DataFrame
    df = pd.DataFrame([data_dict])

    # Ensure all columns exist (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns to match training exactly
    df = df[feature_cols]

    # Predict
    cluster = model.predict(df)[0]
    return SEGMENTS.get(cluster, "Unknown")

def get_prediction_batch(df_raw):
    model, feature_cols = load_resources()
    
    if model is None:
        return None

    # Aggregation Logic
    if 'Trx_Age' not in df_raw.columns or 'Trx_Vlu' not in df_raw.columns:
        return None 

    df_agg = df_raw.groupby('User_Id').agg({
        'Trx_Age': 'min',              
        'Trx_Rank': 'max',             
        'Customer_Age': 'max',         
        'Category In English': 'nunique', 
        'Trx_Vlu': ['sum', 'max'],     
        'Points': 'sum'                
    })

    df_agg.columns = [
        'Recency', 'Frequency', 'Customer_Tenure', 'Category_Diversity',
        'Monetary_Total', 'Monetary_Max', 'Total_Points'
    ]
    df_agg = df_agg.reset_index()

    # Align columns
    X = df_agg.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    clusters = model.predict(X)
    df_agg['Cluster'] = clusters
    df_agg['Segment'] = df_agg['Cluster'].map(SEGMENTS)

    return df_agg