import streamlit as st
import pandas as pd
import os

from inference import get_prediction_manual, get_prediction_batch, SEGMENTS

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="📊")

st.title("Customer Segmentation Tool")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Processing"])

# --- TAB 1: Single Manual Prediction ---
with tab1:
    st.header("Predict Single Customer Segment")
    st.write("Enter the customer's behavioral metrics below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=10)
        frequency = st.number_input("Frequency (Total Transactions)", min_value=1, value=5)
        tenure = st.number_input("Customer Tenure (Days)", min_value=0, value=365)
        diversity = st.number_input("Category Diversity (Unique Categories)", min_value=1, value=2)

    with col2:
        monetary_total = st.number_input("Total Spend ($)", min_value=0.0, value=500.0)
        monetary_max = st.number_input("Max Single Transaction ($)", min_value=0.0, value=100.0)
        points = st.number_input("Loyalty Points", min_value=0, value=50)

    if st.button("Predict Segment"):
        # Create dictionary matching the feature names in train.py
        input_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Customer_Tenure': tenure,
            'Category_Diversity': diversity,
            'Monetary_Total': monetary_total,
            'Monetary_Max': monetary_max,
            'Total_Points': points
        }
        
        result = get_prediction_manual(input_data)
        
        if "Error" in result:
            st.error(result)
        else:
            st.success(f"This customer belongs to: **{result}**")

# --- TAB 2: Batch Processing ---
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a raw transaction CSV (e.g., 'Cleaned_Data_Merchant_Level_2.csv'). The app will aggregate it and segment all customers.")

    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded_file:
        try:
            raw_data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded raw data:", raw_data.head(3))
            
            if st.button("Run Batch Segmentation"):
                with st.spinner("Aggregating data and predicting segments..."):
                    results = get_prediction_batch(raw_data)
                
                if results is not None:
                    st.success("Segmentation Complete!")
                    
                    # Show distribution
                    st.subheader("Segment Distribution")
                    st.bar_chart(results['Segment'].value_counts())
                    
                    st.write("### Customer Results")
                    st.dataframe(results[['User_Id', 'Segment', 'Monetary_Total', 'Frequency']].head())
                    
                    # Download button
                    csv_data = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Segmented Data",
                        data=csv_data,
                        file_name="segmented_customers.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Prediction failed. Please ensure you have trained the model first.")
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")