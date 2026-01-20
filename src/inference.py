import pandas as pd
import joblib
from utils import aggregate_customer_features

SEGMENTS = {
    0: "At-Risk", 1: "One-Timers", 2: "Rising Stars", 3: "Explorers", 4: "VIPs"
}

def load_resources():
    try:
        model = joblib.load('../models/best_model.pkl')
        cols = joblib.load('../models/feature_columns.pkl')
        return model, cols
    except:
        return None, None

def get_prediction_manual(data_dict):
    model, cols = load_resources()
    if not model: return "Model not found"
    
    # Make a DataFrame with 1 row
    df = pd.DataFrame([data_dict])
    
    # Fill missing columns with 0
    for c in cols:
        if c not in df.columns: df[c] = 0
            
    cluster = model.predict(df[cols])[0]
    return SEGMENTS.get(cluster, "Unknown")

def get_prediction_batch(raw_df):
    model, cols = load_resources()
    if not model: return None
    
    # Process raw data
    processed = aggregate_customer_features(raw_df)
    
    # Prepare for model
    X = processed.copy()
    for c in cols:
        if c not in X.columns: X[c] = 0
            
    # Predict
    processed['Cluster'] = model.predict(X[cols])
    processed['Segment'] = processed['Cluster'].map(SEGMENTS)
    
    return processed