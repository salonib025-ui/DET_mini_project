import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset


df = pd.read_csv("../cleaned_data.csv")
print("FILE LOADED SUCCESSFULLY")
print(df.head())
print(df.columns)

# Create Total Price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Group by Customer
customer_data = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'Quantity': 'sum'
}).reset_index()

customer_data.columns = ['CustomerID', 'TotalSpending', 'Transactions', 'TotalItems']

# Feature scaling
X = customer_data[['TotalSpending', 'Transactions', 'TotalItems']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters
def label_cluster(c):
    if c == 0:
        return "Low Value"
    elif c == 1:
        return "Medium Value"
    else:
        return "High Value"

customer_data['Segment'] = customer_data['Cluster'].apply(label_cluster)

# Save output
customer_data.to_csv("../customer_segments.csv", index=False)

# Visualization
plt.scatter(customer_data['TotalSpending'], customer_data['Transactions'],
            c=customer_data['Cluster'])
plt.xlabel("Total Spending")
plt.ylabel("Transactions")
plt.title("Customer Segmentation")
plt.show()

print("✅ Customer segmentation completed!")