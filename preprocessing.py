import pandas as pd
import joblib

# create pickel files so load data quickly
try:
  Addr_Addr = joblib.load("../Data3010csvs/cache/AddrAddr_edgelist.pkl")
  Addr_Tx = joblib.load("../Data3010csvs/cache/AddrTx_edgelist.pkl")
  Tx_Addr = joblib.load("../Data3010csvs/cache/TxAddr_edgelist.pkl")
  Tx_Tx = joblib.load("../Data3010csvs/cache/txs_edgelist.pkl")

  Tx_features = joblib.load("../Data3010csvs/cache/txs_features.pkl")
  Tx_labels = joblib.load("../Data3010csvs/cache/txs_classes.pkl")
  Wallet_features = joblib.load("../Data3010csvs/cache/wallets_features.pkl")
  Wallet_labels = joblib.load("../Data3010csvs/cache/wallets_classes.pkl")
  print("Loaded from cache")

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

  print("Loaded from CSV and cached")




print("Elliptic Plus Files:")
print("Wallet-to-Wallet edges: " + str(Addr_Addr.shape))
print("Wallet-to-Transaction edges: " + str(Addr_Tx.shape))
print("Transaction-to-Wallet edges: " + str(Tx_Addr.shape))
print("Transaction-to-Transaction edges: " + str(Tx_Tx.shape))

print("Transaction Features: " + str(Tx_features.shape))
print("Transaction Labels: " + str(Tx_labels.shape))
print("Wallet Features: " + str(Wallet_features.shape))
print("Wallet Labels: " + str(Wallet_labels.shape))









# edges2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_edgelist.csv")

# trans_features2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_features.csv")

# labels2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_classes.csv")



# print("\nElliptic Original Files:")
# print(edges2.shape)
# print(trans_features2.shape)
# print(labels2.shape)

print(Tx_Tx)
print(Tx_features)

print(Wallet_labels)

