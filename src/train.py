import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
from sklearn.cluster import KMeans

# Set up paths to work from anywhere
FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
DATA_PATH = ROOT_DIR / 'data' / 'raw' / 'Cleaned_Data_Merchant_Level_2.csv'
MODELS_DIR = ROOT_DIR / 'models'

# Add src to path so we can import internal modules
sys.path.append(str(FILE_DIR))
from utils import aggregate_customer_features
from data_pipeline import get_preprocessing_pipeline

def train_model():
    print("--- Starting Training Process ---")
    
    # 1. Load Data
    if not DATA_PATH.exists():
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print("Loading raw data...")
    df_raw = pd.read_csv(DATA_PATH)
    
    # 2. Feature Engineering
    print("Aggregating customer features...")
    df_processed = aggregate_customer_features(df_raw)
    
    # Define features used for clustering
    features_list = [
        'Recency', 'Frequency', 'Customer_Tenure', 'Category_Diversity', 
        'Monetary_Total', 'Monetary_Max', 'Total_Points'
    ]
    
    X = df_processed[features_list].fillna(0)
    
    # 3. Preprocessing
    print("Applying preprocessing pipeline...")
    pipeline = get_preprocessing_pipeline()
    X_scaled = pipeline.fit_transform(X)
    
    # 4. Train Model (KMeans)
    print("Training KMeans model...")
    # Using 5 clusters as defined in your project
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 5. Cluster Sorting 
    monetary_idx = features_list.index('Monetary_Total')
    
    # Get the cluster centers and sort them by Monetary Value
    sorted_idx = np.argsort(kmeans.cluster_centers_[:, monetary_idx])
    
    # Create a lookup map {old_label: new_label}
    lookup = np.zeros_like(sorted_idx)
    lookup[sorted_idx] = np.arange(5)
    
    # Reorder the labels and centers
    kmeans.labels_ = lookup[kmeans.labels_]
    kmeans.cluster_centers_ = kmeans.cluster_centers_[sorted_idx]
    
    print("Clusters sorted by Monetary Value.")

    # 6. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Saving models to", MODELS_DIR)
    joblib.dump(pipeline, MODELS_DIR / 'preprocessing_pipeline.pkl')
    joblib.dump(kmeans, MODELS_DIR / 'best_model.pkl')
    joblib.dump(features_list, MODELS_DIR / 'feature_columns.pkl')
    
    print("--- Training Complete! ---")

if __name__ == "__main__":
    train_model()