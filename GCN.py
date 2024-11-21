import pandas as pd
import joblib

# create pickel files to load data quickly
try:
  Addr_Addr = joblib.load("../Data3010csvs/cache/AddrAddr_edgelist.pkl")
  # Addr_Tx = joblib.load("../Data3010csvs/cache/AddrTx_edgelist.pkl")
  # Tx_Addr = joblib.load("../Data3010csvs/cache/TxAddr_edgelist.pkl")
  # Tx_Tx = joblib.load("../Data3010csvs/cache/txs_edgelist.pkl")

  # Tx_features = joblib.load("../Data3010csvs/cache/txs_features.pkl")
  # Tx_labels = joblib.load("../Data3010csvs/cache/txs_classes.pkl")
  Wallet_features = joblib.load("../Data3010csvs/cache/wallets_features.pkl")
  Wallet_labels = joblib.load("../Data3010csvs/cache/wallets_classes.pkl")
  print("Loaded from cache\n")

except FileNotFoundError:

  Addr_Addr = pd.read_csv("../Data3010csvs/eplus/AddrAddr_edgelist.csv")
  Addr_Tx = pd.read_csv("../Data3010csvs/eplus/AddrTx_edgelist.csv")
  Tx_Addr = pd.read_csv("../Data3010csvs/eplus/TxAddr_edgelist.csv")
  Tx_Tx = pd.read_csv("../Data3010csvs/eplus/txs_edgelist.csv")

  Tx_features = pd.read_csv("../Data3010csvs/eplus/txs_features.csv")
  Tx_labels = pd.read_csv("../Data3010csvs/eplus/txs_classes.csv")
  Wallet_features = pd.read_csv("../Data3010csvs/eplus/wallets_features.csv")
  Wallet_labels = pd.read_csv("../Data3010csvs/eplus/wallets_classes.csv")

  joblib.dump(Addr_Addr, "../Data3010csvs/cache/AddrAddr_edgelist.pkl")
  joblib.dump(Addr_Tx, "../Data3010csvs/cache/AddrTx_edgelist.pkl")
  joblib.dump(Tx_Addr, "../Data3010csvs/cache/TxAddr_edgelist.pkl")
  joblib.dump(Tx_Tx, "../Data3010csvs/cache/txs_edgelist.pkl")

  joblib.dump(Tx_features, "../Data3010csvs/cache/txs_features.pkl")
  joblib.dump(Tx_labels, "../Data3010csvs/cache/txs_classes.pkl")
  joblib.dump(Wallet_features, "../Data3010csvs/cache/wallets_features.pkl")
  joblib.dump(Wallet_labels, "../Data3010csvs/cache/wallets_classes.pkl")

  print("Loaded from CSV and cached\n")

print("Elliptic Plus Files:\n")
print("Wallet-to-Wallet edges: " + str(Addr_Addr.shape))
# print("Wallet-to-Transaction edges: " + str(Addr_Tx.shape))
# print("Transaction-to-Wallet edges: " + str(Tx_Addr.shape))
# print("Transaction-to-Transaction edges: " + str(Tx_Tx.shape))

# print("Transaction Features: " + str(Tx_features.shape))
# print("Transaction Labels: " + str(Tx_labels.shape))
print("Wallet Features: " + str(Wallet_features.shape))
print("Wallet Labels: " + str(Wallet_labels.shape))


print(Wallet_labels)



RANDOM_STATE = 3010

# figure out how to balance imbalance dataset first?

# makes no difference in model perforamnce
# features_first = Wallet_features.drop_duplicates(subset="address", keep="first")
features = Wallet_features.drop_duplicates(subset="address", keep="last")


labels_and_features = pd.merge(Wallet_labels, features, how="inner", on="address")


import torch
from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch_geometric.loader import DataLoader

labels_and_features["node_id"] = range(1, len(labels_and_features) + 1)
print(labels_and_features)

address_to_id = labels_and_features.set_index('address')['node_id'].to_dict()

# 1 = illicit
# 0 = licit
# -1 = unknown
labels_and_features["class"] = labels_and_features["class"].map({2:0, 3:-1})

node_and_features = labels_and_features.iloc[:1000]

# node_features = torch.tensor(labels_and_features[['feature_1', 'feature_2', 'feature_3']].values, dtype=torch.float)
# node_labels = torch.tensor(labels_and_features['label'].values, dtype=torch.long)



# Addr_Addr["input_address"] = Addr_Addr["input_address"].map(address_to_id)
# Addr_Addr["output_address"] = Addr_Addr["output_address"].map(address_to_id)
# print(Addr_Addr)

# print(len(Addr_Addr.loc[Addr_Addr["input_address"].isna()]))

# edges = Addr_Addr.loc[:10000]

# transpose_edges = edges[["input_address", "output_address"]].values.T

# edge_index = torch.tensor(transpose_edges, dtype = torch.long)




# # Step 3: Create the PyTorch Geometric Data Object
# data = Data(x=node_features, edge_index=edge_index, y=node_labels)
# print(data)
