import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../src'))

from inference import get_prediction_manual, get_prediction_batch, SEGMENTS

st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #0e1117;}
    div.stButton > button {width: 100%; background-color: #ff4b4b; color: white;}
</style>
""", unsafe_allow_html=True)

st.title("Customer Segmentation Tool")

@st.cache_data
def load_dashboard_data():
    try:
        data_path = os.path.join(current_dir, '../data/preprocessed/preprocessed_df.parquet')
        df = pd.read_parquet(data_path)
        df['Segment'] = df['Cluster'].map(SEGMENTS)
        return df
    except FileNotFoundError:
        return None

tab1, tab2, tab3 = st.tabs(["Dashboard Analysis", "Single Prediction", "Batch Processing"])

with tab1:
    df = load_dashboard_data()
    
    if df is not None:
        st.markdown("### Key Performance Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{len(df):,}")
        m2.metric("Avg Spend", f"${df['Monetary_Total'].mean():.0f}")
        m3.metric("Avg Frequency", f"{df['Frequency'].mean():.1f}")
        m4.metric("Avg Tenure", f"{df['Customer_Tenure'].mean():.0f} days")
        
        st.divider()

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Distribution")
            fig_pie = px.pie(df, names='Segment', values='User_Id', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Revenue by Segment")
            revenue_data = df.groupby('Segment')['Monetary_Total'].sum().reset_index()
            fig_bar = px.bar(revenue_data, x='Segment', y='Monetary_Total', color='Segment',
                             text_auto='.2s', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_bar.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Total Revenue")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Segment Profiles")
        
        segment_stats = df.groupby('Segment')[['Monetary_Total', 'Frequency', 'Recency', 'Customer_Tenure']].mean()
        
        st.dataframe(segment_stats.style.format("{:.1f}").background_gradient(cmap="Blues"), 
                    use_container_width=True)

    else:
        st.warning("No data found. Please run the training script first.")

with tab2:
    st.markdown("### Predict Customer Segment")
    st.info("Enter customer data below to predict their segment.")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            recency = st.number_input("Recency (days since last transaction)", min_value=0, value=10)
            frequency = st.number_input("Frequency (total transactions)", min_value=1, value=5)
            tenure = st.number_input("Tenure (days as customer)", min_value=0, value=365)
            diversity = st.number_input("Category Diversity", min_value=1, value=2)

        with col_b:
            total_spend = st.number_input("Total Spend", min_value=0.0, value=1000.0)
            max_spend = st.number_input("Max Single Transaction", min_value=0.0, value=200.0)
            points = st.number_input("Loyalty Points", min_value=0, value=50)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Predict Segment")

    if submit:
        customer_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Customer_Tenure': tenure,
            'Category_Diversity': diversity,
            'Monetary_Total': total_spend,
            'Monetary_Max': max_spend,
            'Total_Points': points
        }
        
        predicted_segment = get_prediction_manual(customer_data)
        
        st.success(f"Predicted Segment: **{predicted_segment}**")
        
        if predicted_segment == "VIPs":
            st.balloons()
            st.markdown("**Recommendation:** Send exclusive offers and early access")
        elif predicted_segment == "At-Risk":
            st.markdown("**Recommendation:** Consider sending reactivation offers")

with tab3:
    st.markdown("### Batch Prediction")
    st.write("Upload your raw transaction CSV file for bulk prediction.")

    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

    if uploaded_file:
        try:
            raw_data = pd.read_csv(uploaded_file)
            st.write("Preview:", raw_data.head(3))
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    results = get_prediction_batch(raw_data)
                
                if results is not None:
                    st.success("Prediction Complete")
                    
                    st.write("### Results")
                    st.dataframe(results[['User_Id', 'Segment', 'Monetary_Total', 'Frequency']].head())
                    
                    csv_data = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results",
                        data=csv_data,
                        file_name="segmented_customers.csv",
                        mime="text/csv"
                    )
                    
                    st.bar_chart(results['Segment'].value_counts())
                    
        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please ensure your CSV has the required columns")