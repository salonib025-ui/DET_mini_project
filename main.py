import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("cleaned_data.csv")
rules = pd.read_csv("association_rules.csv")
segments = pd.read_csv("customer_segments.csv")

# ==============================
# BASIC INFO
# ==============================

print("Dataset shape:", data.shape)

print("\nTop Association Rules:")
print(rules.head())

print("\nCustomer Segments:")
print(segments.head())


# ==============================
# 1. SCATTER PLOT (Customer Segmentation)
# ==============================

plt.figure(figsize=(8,6))

plt.scatter(
    segments['TotalSpending'],
    segments['Transactions'],
    c=segments['Cluster']
)

plt.xlabel("Total Spending")
plt.ylabel("Number of Transactions")
plt.title("Customer Segmentation")

plt.grid(True)

# Save image for PPT
plt.savefig("scatter_plot.png")

plt.show()


# ==============================
# 2. CORRELATION HEATMAP
# ==============================

# Create TotalPrice (fix for missing column)
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Select numerical columns
heatmap_data = data[['Quantity', 'UnitPrice', 'TotalPrice']]

# Correlation matrix
corr = heatmap_data.corr()

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

# Save image for PPT
plt.savefig("heatmap.png")

plt.show()

# Print correlation values
print("\nCorrelation Matrix:\n", corr)


# ==============================
# 3. OPTIONAL: HISTOGRAM (Extra Marks)
# ==============================

plt.figure()

data['TotalPrice'].hist()

plt.title("Distribution of Total Price")
plt.xlabel("Total Price")
plt.ylabel("Frequency")

plt.show()