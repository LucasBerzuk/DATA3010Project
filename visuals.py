import pandas as pd
import joblib
import random

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors


# create pickel files to load data quickly
try:
  Addr_Addr = joblib.load("../Data3010csvs/cache/AddrAddr_edgelist.pkl")
  Wallet_features = joblib.load("../Data3010csvs/cache/wallets_features.pkl")
  Wallet_labels = joblib.load("../Data3010csvs/cache/wallets_classes.pkl")
  print("Loaded from cache\n")

except FileNotFoundError:

  Addr_Addr = pd.read_csv("../Data3010csvs/eplus/AddrAddr_edgelist.csv")
  Wallet_features = pd.read_csv("../Data3010csvs/eplus/wallets_features.csv")
  Wallet_labels = pd.read_csv("../Data3010csvs/eplus/wallets_classes.csv")

  joblib.dump(Addr_Addr, "../Data3010csvs/cache/AddrAddr_edgelist.pkl")
  joblib.dump(Wallet_features, "../Data3010csvs/cache/wallets_features.pkl")
  joblib.dump(Wallet_labels, "../Data3010csvs/cache/wallets_classes.pkl")

  print("Loaded from CSV and cached\n")


# Step 1: Create a graph
G = nx.Graph()

# Step 2: Add nodes with features ---------------------------------------ALL NODES-----------------------------------------------
# print("Adding nodes please wait...")
# nodes_data = Wallet_features[['address']].join(Wallet_features.drop('address', axis=1))  # Join address with features
# nodes_array = nodes_data.to_numpy()
# # Add nodes in bulk
# for i, row in enumerate(nodes_array):
#     wallet_id = row[0]  # 'address' column is the first column
#     features = {Wallet_features.columns[j]: row[j] for j in range(1, len(row))}  # Mapping features to dictionary
#     G.add_node(wallet_id, **features) 
# print(f"Total number of nodes: {G.number_of_nodes()}")

# Step 2: Add nodes (sampling)
print("Adding nodes please wait...")
nodes_data = Wallet_features[['address']].join(Wallet_features.drop('address', axis=1))  # Join address with features
nodes_array = nodes_data.to_numpy()
num_nodes_to_sample = int(len(nodes_array) * 0.0005)  # .05% of total nodes
sampled_nodes = random.sample(list(nodes_array), num_nodes_to_sample)
for row in sampled_nodes:
    wallet_id = row[0]  # 'address' column is the first column
    features = {Wallet_features.columns[j]: row[j] for j in range(1, len(row))}  # Mapping features to dictionary
    G.add_node(wallet_id, **features)
print(f"Total number of nodes added: {G.number_of_nodes()}")

# Step 3: Add edges--------------------------------------------ALL EDGES----------------------------------------------------
# print("Adding edges please wait...")
# edges = Addr_Addr[['input_address', 'output_address']].to_numpy()
# for i, (source, target) in enumerate(edges):
#     G.add_edge(source, target)
# print(f"Total number of edges: {G.number_of_edges()}")

# Step 3: Add edges that exist from the nodes sampled
print("Adding edges please wait...")
existing_nodes = set(G.nodes)
edges = Addr_Addr[['input_address', 'output_address']].to_numpy()
filtered_edges = [(source, target) for source, target in edges if source in existing_nodes and target in existing_nodes]
G.add_edges_from(filtered_edges)
print(f"Total number of edges added: {G.number_of_edges()}")

# Step 4: Visualize the graph
print("Setting figure size")
plt.figure(figsize=(12, 8))

# Create
print("Creating layout")
pos = nx.spiral_layout(G)
# Other layouts that work 'spring_layout' 'shell_' 'circular_' 'random_'

# Draw nodes with a color map based on a specific feature (e.g., 'btc_transacted_total')
print("Setting node colour")
node_color = [data['btc_transacted_total'] for _, data in G.nodes(data=True)]  # Adjust the feature name as needed
node_size = 5  # Adjust the size if needed

# Draw nodes (only once to improve performance)
print("Drawing nodes")
nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, node_size=node_size)

# Draw edges with reduced alpha (to prevent cluttering)
print("Drawing edges in chunks") # This is pointless now because i sampled the nodes instead of doing all
edges = list(G.edges())
chunk_size = 5000  # Number of edges to draw per chunk
for i in range(0, len(edges), chunk_size):
    chunk = edges[i:i + chunk_size]
    nx.draw_networkx_edges(G, pos, edgelist=chunk, alpha=0.3)
    print(f"Drawn edges {i} to {i + chunk_size}")

print("Adding title and axis")
plt.title("Wallet Feature Graph")
plt.axis('off')  # Hide axes for cleaner visualization
print("Showing graph")
plt.show()











# This is still important for if i do all nodes
# Sample a fraction of edges to draw (e.g., 10% of edges)
# num_edges_to_draw = int(len(G.edges) * 0.05)  # Adjust the fraction as needed
# sampled_edges = random.sample(list(G.edges()), num_edges_to_draw)

# print("Drawing sampled edges")
# nx.draw_networkx_edges(G, pos, edgelist=sampled_edges, alpha=0.3)



# Optionally add labels for only a subset of nodes (or disable if the graph is too large)
# You can avoid drawing labels for large graphs to save computation time
# Here we draw labels only for a smaller subset of nodes
# print("Adding labels")
# labels = {node: node for node, data in G.nodes(data=True) if node in list(G.nodes)[:50]}  # Show labels for first 50 nodes
# nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

# Add color bar based on the feature (e.g., 'btc_transacted_total')
# print("Adding colour bar")
# plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='BTC Transacted Total')
# node_color = [data['btc_transacted_total'] for _, data in G.nodes(data=True)]  # Replace with actual feature
# norm = mcolors.Normalize(vmin=min(node_color), vmax=max(node_color))
# sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
# sm.set_array([])  # Set the array to an empty one; not used but required for ScalarMappable
# plt.colorbar(sm, label='BTC Transacted Total')






# # Step 4: Visualize the graph
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G)  # Generate positions for nodes

# # Draw nodes with a color map based on a specific feature (e.g., feature_1)
# node_color = [data['btc_transacted_total'] for _, data in G.nodes(data=True)]  # Replace 'feature_1' with a feature name
# nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, node_size=50)

# # Draw edges
# nx.draw_networkx_edges(G, pos, alpha=0.5)

# # Add labels (optional, can be cluttered for large graphs)
# nx.draw_networkx_labels(G, pos, font_size=8)

# plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Feature 1 Value')
# plt.title("Wallet Feature Graph")
# plt.show()
