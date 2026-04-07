import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load and Prepare Data
df = pd.read_csv("transactions_data.csv")
df['Items'] = df['Items'].apply(eval)
transactions = df['Items'].tolist()

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_data, columns=te.columns_)

# 2. Mining Rules
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filtering for clarity
rules = rules[(rules['lift'] > 2) & (rules['confidence'] > 0.3)]
rules = rules.sort_values(by='lift', ascending=False).head(15)

# 3. DEFINE 'G' BEFORE USING IT (This fixes your NameError)
G = nx.DiGraph()

for _, row in rules.iterrows():
    for a in row['antecedents']:
        for c in row['consequents']:
            G.add_edge(a, c, weight=row['lift'])

# 4. Visualization logic
plt.figure(figsize=(14, 10))

# 1. Standard layout with k=1.5
pos = nx.spring_layout(G, k=1.5, seed=42) 

edges = G.edges(data=True)
# Keep edge weights normalized for visibility
weights = [d['weight'] / rules['lift'].max() * 6 for (_, _, d) in edges]

# Draw Nodes and Labels
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.9)

# 2. Add node labels (cleaned)
# Iterate through the nodes and make sure the text is clean (no tuples)
node_labels = {node: str(node) for node in G.nodes()} # Clean any non-string keys
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

# 3. Add curved edges
# This prevents overlap for bidirectional rules
curved_edges = [edge for edge in edges] # We need to use nx.draw_networkx_edges for connectionstyle
nx.draw_networkx_edges(
    G, pos, 
    width=weights, 
    edge_color='gray', 
    alpha=0.6, 
    arrows=True, 
    arrowsize=15,
    connectionstyle="arc3,rad=0.2" 
)

# 4. EXPLICITLY ADD EDGE LABELS
# We want to label the 'weight' (which is lift) on the edges.
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=7)

plt.title("Filtered Association Rules Network", fontsize=15)
plt.axis('off')
plt.show()