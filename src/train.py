import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import your helper function
from utils import aggregate_customer_features

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Cleaned_Data_Merchant_Level_2.csv') 
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model():
    print("1. Loading raw data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
        
    df_raw = pd.read_csv(DATA_PATH)
    
    print("2. Aggregating data (Feature Engineering)...")
    df_processed = aggregate_customer_features(df_raw)
    
    # Select features for training
    features = [
        'Recency', 'Frequency', 'Customer_Tenure', 'Category_Diversity', 
        'Monetary_Total', 'Monetary_Max', 'Total_Points'
    ]
    
    # Ensure columns exist and fill NaNs
    X = df_processed[features].fillna(0)

    print("3. Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("4. Training KMeans model...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)

    # Sort clusters: 0 = Lowest Value, 4 = Highest Value based on Total Spend (Monetary_Total)
    monetary_idx = features.index('Monetary_Total')
    
    # Sort cluster centers based on Monetary Value
    idx = np.argsort(kmeans.cluster_centers_[:, monetary_idx])
    
    # Create a lookup table to remap labels
    lookup = np.zeros_like(idx)
    lookup[idx] = np.arange(5)
    
    # Reorder labels
    kmeans.labels_ = lookup[kmeans.labels_]
    kmeans.cluster_centers_ = kmeans.cluster_centers_[idx]

    print("5. Saving models...")
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(features, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    
    print("Training complete! Models saved in 'models/' folder.")

if __name__ == "__main__":
    train_model()