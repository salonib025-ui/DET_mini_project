import pandas as pd

# 1. Load dataset
df = pd.read_csv("online_retail.csv")

# 2. Data Cleaning
df = df.dropna(subset=['Description'])     # remove missing descriptions
df = df[df['Quantity'] > 0]                # remove returns/invalid entries
df = df.drop_duplicates()                 # remove duplicates

# 3. Feature Engineering
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 4. Create Transactions
transactions = df.groupby('InvoiceNo')['Description'].apply(list)

# Convert transactions to DataFrame (so it can be saved)
transactions_df = transactions.reset_index()
transactions_df.columns = ['InvoiceNo', 'Items']

# 5. Save outputs for team use
df.to_csv("cleaned_retail_data.csv", index=False)
transactions_df.to_csv("transactions_data.csv", index=False)

print("✅ Preprocessing complete!")
print("Files created:")
print("1. cleaned_retail_data.csv")
print("2. transactions_data.csv")