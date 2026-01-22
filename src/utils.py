import pandas as pd

def aggregate_customer_features(df):
    """
    Aggregates raw transaction data into a customer-level profile.
    """
    # Group by User to get Recency, Frequency, Monetary, etc.
    customer_features = df.groupby('User_Id').agg({
        'Trx_Age': 'min',              # Recency (Days since last purchase)
        'Trx_Rank': 'max',             # Frequency (Total transactions)
        'Trx_Vlu': ['sum', 'max'],     # Monetary Total & Max Spend
        'Points': 'sum',               # Total Loyalty Points
        'Customer_Age': 'max'          # Tenure (Days since first signup)
    }).reset_index()
    
    # Flatten MultiIndex columns
    customer_features.columns = [
        'User_Id', 'Recency', 'Frequency', 'Monetary_Total', 
        'Monetary_Max', 'Total_Points', 'Customer_Tenure'
    ]
    
    # Calculate Diversity (Number of unique categories purchased)
    if 'Category In English' in df.columns:
        diversity = df.groupby('User_Id')['Category In English'].nunique().reset_index()
        diversity.columns = ['User_Id', 'Category_Diversity']
        # Merge diversity into main features
        final_df = customer_features.merge(diversity, on='User_Id', how='left')
    else:
        # If category column missing, fill 1
        final_df = customer_features
        final_df['Category_Diversity'] = 1
    
    # Fill any missing values with 0
    return final_df.fillna(0)