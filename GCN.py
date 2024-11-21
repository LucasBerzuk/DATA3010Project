import pandas as pd
import joblib

import os

# Set environment variable for the current session
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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

labels_and_features["node_id"] = range(0, len(labels_and_features))
print(labels_and_features)

address_to_id = labels_and_features.set_index('address')['node_id'].to_dict()

# 1 = illicit
# 0 = licit
# -1 = unknown
labels_and_features["class"] = labels_and_features["class"].map({2:0, 3:-1})

# node_and_features = labels_and_features.iloc[:100]
node_and_features = labels_and_features.loc[labels_and_features["node_id"].isin(range(1,101))]

node_features = node_and_features.drop(columns=["class", "address", "Time step", "total_txs"])

# print(node_and_features.columns)

node_features = torch.tensor(node_features.values, dtype=torch.float)
node_labels = torch.tensor(node_and_features["class"].values, dtype=torch.long)


Addr_Addr["input_address"] = Addr_Addr["input_address"].map(address_to_id)
Addr_Addr["output_address"] = Addr_Addr["output_address"].map(address_to_id)
# print(Addr_Addr)

# print(len(Addr_Addr.loc[Addr_Addr["input_address"].isna()]))

print(node_and_features["node_id"])

edges = Addr_Addr.loc[Addr_Addr["input_address"].isin(node_and_features["node_id"]) & Addr_Addr["output_address"].isin(node_and_features["node_id"])]



transpose_edges = edges[["input_address", "output_address"]].values.T


edge_index = torch.tensor(transpose_edges, dtype = torch.long)

data = Data(x=node_features, edge_index=edge_index, y=node_labels)
print(data)



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        
        # Define the two layers for the GCN
        self.conv1 = GCNConv(in_channels, 16)  # First layer (input -> hidden)
        self.conv2 = GCNConv(16, out_channels)  # Second layer (hidden -> output)
    
    def forward(self, data):
        # Forward pass
        x, edge_index = data.x, data.edge_index
        
        # Apply first convolution layer followed by ReLU activation and dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        
        # Apply second convolution layer
        x = self.conv2(x, edge_index)
        
        return x  # Raw output logits (before applying softmax)
    

train_mask = torch.rand(len(data.x)) < 0.8  # 80% for training
val_mask = ~train_mask
test_mask = ~train_mask


import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score

# Initialize the model, optimizer, and loss function
model = GCN(in_channels=node_features.shape[1], out_channels=2)  # Binary classification (illicit/licit)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss is commonly used for classification

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data)
    
    # Compute loss using only the training mask (ignore test/val data)
    loss = criterion(out[train_mask], data.y[train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluation loop
def evaluate(mask):
    model.eval()
    out = model(data)
    
    # Apply softmax to get class probabilities
    pred = out.argmax(dim=1)
    
    # Compute accuracy for the given mask (train/val/test)
    acc = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
    return acc

# Training the model
for epoch in range(100):  # Train for 100 epochs
    loss = train()
    if epoch % 10 == 0:
        val_acc = evaluate(val_mask)  # Evaluate on validation set
        print(f'Epoch {epoch}: Loss = {loss:.4f}, Val Accuracy = {val_acc:.4f}')
    
# After training, evaluate on the test set
test_acc = evaluate(test_mask)
print(f'Test Accuracy: {test_acc:.4f}')