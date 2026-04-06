import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load transactions file
df = pd.read_csv("transactions_data.csv")

# Convert string list → actual list
df['Items'] = df['Items'].apply(eval)

transactions = df['Items'].tolist()

# 2. Convert into basket format
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_data, columns=te.columns_)

# 3. Apply Apriori Algorithm
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

# 4. Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 5. Save outputs
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)

print("✅ Apriori completed!")
print("Files created:")
print("1. frequent_itemsets.csv")
print("2. association_rules.csv")