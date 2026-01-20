import pandas as pd

def aggregate_customer_features(df):
    # Group by User to get Recency, Frequency, Monetary, etc.
    customer_features = df.groupby('User_Id').agg({
        'Trx_Age': 'min',              # Recency
        'Trx_Rank': 'max',             # Frequency
        'Trx_Vlu': ['sum', 'max'],     # Monetary Total & Max
        'Points': 'sum',               # Total Points
        'Customer_Age': 'max'          # Tenure
    }).reset_index()
    
    # Clean up column names
    customer_features.columns = ['User_Id', 'Recency', 'Frequency', 'Monetary_Total', 'Monetary_Max', 'Total_Points', 'Customer_Tenure']
    
    # Calculate Diversity (Unique Categories)
    diversity = df.groupby('User_Id')['Category In English'].nunique().reset_index()
    diversity.columns = ['User_Id', 'Category_Diversity']
    
    # Merge everything together
    final_df = customer_features.merge(diversity, on='User_Id', how='left')
    
    return final_df