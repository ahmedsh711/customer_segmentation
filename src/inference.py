import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# UPDATED: 5 Segment Names
SEGMENT_LABELS = {
    0: 'Murky (Inactive)',
    1: 'Bronze (Low Value)',
    2: 'Silver (Average)',
    3: 'Gold (High Potential)',
    4: 'Diamond (VIP)'
}

def make_prediction(input_data):
    try:
        # Load files
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

        # Prepare data
        df = pd.DataFrame([input_data])
        df = df[feature_cols]

        # Scale data
        df_scaled = scaler.transform(df)

        # Predict
        cluster_id = model.predict(df_scaled)[0]
        
        return SEGMENT_LABELS.get(cluster_id, f"Cluster {cluster_id}")

    except Exception as e:
        return f"Error: {str(e)}"