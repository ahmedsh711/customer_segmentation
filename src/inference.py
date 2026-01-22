import pandas as pd
import joblib
from pathlib import Path

# Setup Paths
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
MODELS_DIR = ROOT_DIR / 'models'

# Segment Names mapped to sorted clusters (0 = Lowest, 4 = Highest)
SEGMENT_NAMES = {
    0: 'Murky (Inactive)',
    1: 'Bronze (Low Value)',
    2: 'Silver (Average)',
    3: 'Gold (High Potential)',
    4: 'Diamond (VIP)'
}

def load_artifacts():
    try:
        pipeline = joblib.load(MODELS_DIR / 'preprocessing_pipeline.pkl')
        model = joblib.load(MODELS_DIR / 'best_model.pkl')
        features = joblib.load(MODELS_DIR / 'feature_columns.pkl')
        return model, pipeline, features
    except FileNotFoundError:
        return None, None, None

def make_prediction(input_data):
    model, pipeline, feature_cols = load_artifacts()
    
    if not model:
        return "Error: Model files not found. Please train the model first."
        
    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure correct column order
    df = df[feature_cols]
    
    # Preprocess
    data_scaled = pipeline.transform(df)
    
    # Predict
    cluster_id = model.predict(data_scaled)[0]
    
    return SEGMENT_NAMES.get(cluster_id, "Unknown")

def make_batch_prediction(df_customers):
    model, pipeline, feature_cols = load_artifacts()
    
    if not model:
        return None
    
    # Ensure columns exist and order matches
    try:
        data_for_pred = df_customers[feature_cols].fillna(0)
    except KeyError as e:
        return f"Error: Missing columns in uploaded data. Expected: {feature_cols}"

    # Preprocess
    data_scaled = pipeline.transform(data_for_pred)
    
    # Predict
    clusters = model.predict(data_scaled)
    
    # Add results to original dataframe
    df_customers['Cluster_ID'] = clusters
    df_customers['Segment'] = df_customers['Cluster_ID'].map(SEGMENT_NAMES)
    
    return df_customers