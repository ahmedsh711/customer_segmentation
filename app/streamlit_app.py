import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / 'src'))

from inference import make_prediction, make_batch_prediction
from utils import aggregate_customer_features

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation System")
st.markdown("Predict customer segments using AI (K-Means Clustering).")

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose Mode:", ["Manual Prediction", "Batch Prediction (CSV)"])

if option == "Manual Prediction":
    st.header("Single Customer Prediction")
    st.info("Enter the customer's aggregated metrics below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.number_input("Recency (Days since last visit)", min_value=0, value=10)
        frequency = st.number_input("Frequency (Total Transactions)", min_value=1, value=5)
        tenure = st.number_input("Tenure (Days since signup)", min_value=0, value=365)
        diversity = st.number_input("Category Diversity (Unique Categories)", min_value=1, value=2)

    with col2:
        monetary_total = st.number_input("Total Spend ($)", min_value=0.0, value=500.0)
        monetary_max = st.number_input("Max Single Receipt ($)", min_value=0.0, value=100.0)
        points = st.number_input("Total Loyalty Points", min_value=0, value=50)

    if st.button("Predict Segment"):
        input_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Customer_Tenure': tenure,
            'Category_Diversity': diversity,
            'Monetary_Total': monetary_total,
            'Monetary_Max': monetary_max,
            'Total_Points': points
        }
        
        result = make_prediction(input_data)
        
        if "Error" in result:
            st.error(result)
        else:
            st.success(f"**Customer Segment:** {result}")
            
            # Simple Description Logic
            if "VIP" in result:
                st.balloons()
                st.markdown(" **Action:** Treat with exclusivity. Offer concierge service.")
            elif "Inactive" in result:
                st.markdown(" **Action:** Send win-back campaign immediately.")
            elif "Potential" in result:
                st.markdown(" **Action:** Offer loyalty rewards to push to VIP.")

elif option == "Batch Prediction (CSV)":
    st.header("Batch Processing")
    st.write("Upload a raw transaction CSV (must have `User_Id`, `Trx_Vlu`, etc.) or pre-aggregated data.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        
        if st.button("Process & Predict"):
            # Check if it looks like raw data or aggregated data
            if 'Trx_Vlu' in df.columns and 'User_Id' in df.columns:
                st.info("Detected Raw Transaction Data. Aggregating first...")
                df_processed = aggregate_customer_features(df)
            else:
                st.info("Assuming Pre-aggregated Data...")
                df_processed = df
            
            results = make_batch_prediction(df_processed)
            
            if isinstance(results, str): # Error message
                st.error(results)
            else:
                st.success("Segmentation Complete!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", len(results))
                col2.metric("VIP Customers", len(results[results['Segment'].str.contains('VIP')]))
                col3.metric("Inactive Customers", len(results[results['Segment'].str.contains('Inactive')]))
                
                # Chart
                fig = px.pie(results, names='Segment', title='Customer Distribution')
                st.plotly_chart(fig)
                
                # Download
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", csv, "segmented_customers.csv", "text/csv")