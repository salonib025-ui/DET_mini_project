import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("cleaned_data.csv")
rules = pd.read_csv("association_rules.csv")
segments = pd.read_csv("customer_segments.csv")

print("Dataset shape:", data.shape)

print("\nTop Association Rules:")
print(rules.head())

print("\nCustomer Segments:")
print(segments.head())

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    segments['TotalSpending'],
    segments['Transactions'],
    c=segments['Cluster'],   # color by cluster
)

plt.xlabel("Total Spending")
plt.ylabel("Number of Transactions")
plt.title("Customer Segmentation")

plt.grid(True)
plt.show()