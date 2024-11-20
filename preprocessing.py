import pandas as pd
import joblib

# create pickel files so load data quickly
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

w_labelled = Wallet_labels.loc[Wallet_labels["class"] != 3]
w_unlabelled = Wallet_labels.loc[Wallet_labels["class"] == 3]

print("Labelled: " + str(w_labelled.shape))
print("Unlabelled: " + str(w_unlabelled.shape))

dist = w_labelled["class"].value_counts()
print(dist)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

RANDOM_STATE = 3010

# figure out how to balance imbalance dataset first?
features = Wallet_features.drop_duplicates(subset="address", keep="first")

labels_features = pd.merge(w_labelled, features, how="inner", on="address")



labels = labels_features["class"]

features = labels_features.drop(columns=["address", "class"])




X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=RANDOM_STATE)

# Automatically compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=y_train)
class_weight_dict = dict(zip(np.unique(labels), class_weights))

# Train Random Forest with class weights
clf = RandomForestClassifier(class_weight=class_weight_dict, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))







#--------------- Dont think we need anymore ----------------------
# edges2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_edgelist.csv")
# trans_features2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_features.csv")
# labels2 = pd.read_csv("../Data3010csvs/e1/elliptic_txs_classes.csv")

# print("\nElliptic Original Files:")
# print(edges2.shape)
# print(trans_features2.shape)
# print(labels2.shape)

# print(Tx_Tx)
# print(Tx_features)

# print(Wallet_labels)

