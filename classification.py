import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np


RANDOM_STATE = 3010


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

print("Elliptic Plus Files:\n")
print("Wallet-to-Wallet edges: " + str(Addr_Addr.shape))
print("Wallet Features: " + str(Wallet_features.shape))
print("Wallet Labels: " + str(Wallet_labels.shape))



# ---------------------------------- CLASS DISTRIBUTION ------------------------------------

w_labelled = Wallet_labels.loc[Wallet_labels["class"] != 3]
w_unlabelled = Wallet_labels.loc[Wallet_labels["class"] == 3]

print("Labelled: " + str(w_labelled.shape))
print("Unlabelled: " + str(w_unlabelled.shape))
dist = w_labelled["class"].value_counts()
print(dist)

# plt.bar(dist.index.astype(str), dist.values, color=["blue", "red"])
plt.bar(dist.index.map({1:"illicit", 2:"licit"}), dist.values, color=["limegreen", "firebrick"], edgecolor='black')

# Add titles and labels
plt.title('Labelled Addresses', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=14, fontweight='bold')

for index, value in enumerate(dist.values):
    plt.text(index, value + 0.05 * max(dist.values), str(value), 
             ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, 300000)
# Show the plot
plt.show()


#_________TRY_______________ BalancedRandomForestClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.utils.class_weight import compute_class_weight


# makes no difference in model perforamnce
# features_first = Wallet_features.drop_duplicates(subset="address", keep="first")
features = Wallet_features.drop_duplicates(subset="address", keep="last")
labels_and_features = pd.merge(w_labelled, features, how="inner", on="address")

# UNKOWN ADDRESSES
unlabelled_with_features = pd.merge(w_unlabelled, features, how="inner", on="address")
unlabelled_features = unlabelled_with_features.drop(columns=["address", "class", "Time step", "num_timesteps_appeared_in"])


labels = labels_and_features["class"]
features = labels_and_features.drop(columns=["address", "class", "Time step", "num_timesteps_appeared_in"])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=RANDOM_STATE)


# ---------------------------------- FEATURE DISTRIBUTION ------------------------------------

# summary = features['btc_transacted_mean'].describe(percentiles=[.25, .5, .75])
# print(summary)
# small_btc_values = features['btc_transacted_mean'][features['btc_transacted_mean'] < 1]

# # plt.boxplot(features['btc_transacted_mean'], vert=False)  # 'vert=False' makes it horizontal
# # plt.title('Boxplot of btc_transacted_mean')
# # plt.xlabel('Bitcoin Amount')
# # plt.show()
# # Plotting the histogram for 'feature_name'
# plt.hist(small_btc_values, bins=100, edgecolor='black')  # Adjust bins for your data range
# plt.title('Histogram of btc_transacted_mean')
# plt.xlabel('Bitcoin Amount')
# plt.ylabel('Frequency')
# plt.show()



# ---------------------------------- BALANCING ------------------------------------

# # Automatically compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=y_train)
class_weight_dict = dict(zip(np.unique(labels), class_weights))

from imblearn.over_sampling import SMOTE

smote_object = SMOTE(random_state=RANDOM_STATE)

X_train_smote, y_train_smote = smote_object.fit_resample(X_train, y_train)
dist = y_train_smote.value_counts()
print(dist)



# ---------------------------------- SCALING ------------------------------------

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler_R = RobustScaler()
X_train_scaleR = scaler_R.fit_transform(X_train)
X_test_scaleR = scaler_R.fit_transform(X_test)
# scaler = MinMaxScaler()
# scaler_N = StandardScaler()

# X_train_scaleNorm = scaler_N.fit_transform(X_train_smote)
# X_test_scaleNorm = scaler_N.fit_transform(X_test)


# ---------------------------------- RESULTS TABLE ------------------------------------

if (os.path.exists("../Data3010csvs/preds/RF.csv")):
  print("RF preds exist")

  RF_pred = pd.read_csv("../Data3010csvs/preds/RF.csv")
  RF_model = joblib.load("../Data3010csvs/models/RF_model.pkl")

else:
  print("Starting training RF\n") 
  RF_model = RandomForestClassifier(#class_weight=class_weight_dict,
                                     random_state=RANDOM_STATE)
  RF_model.fit(X_train, y_train)
  print("Finished training RF\n")

  RF_pred = RF_model.predict(X_test)
  pd.DataFrame({'y_pred': RF_pred}).to_csv("../Data3010csvs/preds/RF.csv", index=False)

  joblib.dump(RF_model, '../Data3010csvs/models/RF_model.pkl')



#------------------------------------ NEW PREDICTIONS -------------------------------

# unlabelled_preds = RF_model.predict(unlabelled_features)
# unlabelled = pd.DataFrame({
#     'address': unlabelled_with_features['address'],  # The 'address' column from your unlabeled data
#     'class': unlabelled_preds  # The predicted classes (e.g., 0 for licit, 1 for illicit)
# })

# dist = unlabelled["class"].value_counts()
# print(dist)
# # change illicit prediction to 3 and licit prediction to 4
# unlabelled["class"] = unlabelled["class"].map({1:3,2:4})

# dist = unlabelled["class"].value_counts()
# print(dist)

# all_addresses = pd.concat([w_labelled, unlabelled], ignore_index=True)
# all_addresses.to_csv("../Data3010csvs/preds/new_preds.csvs")




# ---------------------------------- LR ------------------------------------

from sklearn.linear_model import LogisticRegression

if (os.path.exists("../Data3010csvs/preds/LR.csv")):
  print("LR preds exist")

  LR_pred = pd.read_csv("../Data3010csvs/preds/LR.csv")

else:
  print("Starting training LR\n") 
  LR_model = LogisticRegression(#class_weight=class_weight_dict, 
                                  random_state=RANDOM_STATE)
  LR_model.fit(X_train_smote, y_train_smote)
  print("Finished training LR\n")

  LR_pred = LR_model.predict(X_test)
  pd.DataFrame({'y_pred': LR_pred}).to_csv("../Data3010csvs/preds/LR.csv", index=False)




# ---------------------------------- MLP ------------------------------------

from sklearn.neural_network import MLPClassifier

if (os.path.exists("../Data3010csvs/preds/MLP.csv")):
  print("MLP preds exist")

  MLP_pred = pd.read_csv("../Data3010csvs/preds/MLP.csv")

else:
  print("Starting training MLP\n") 

  MLP_model = MLPClassifier(#class_weight=class_weight_dict, 
                            hidden_layer_sizes=(50,), 
                            solver='adam', #optimizer
                            learning_rate_init = 0.001,
                            max_iter = 200, # match elliptic 1 studies epoch
                            random_state = RANDOM_STATE)


  MLP_model.fit(X_train_scaleR, y_train)
  print("Finished training MLP\n")

  # Predictions and evaluation
  MLP_pred = MLP_model.predict(X_test_scaleR)
  pd.DataFrame({'y_pred': MLP_pred}).to_csv("../Data3010csvs/preds/MLP.csv", index=False)




# ---------------------------------- RESULTS TABLE ------------------------------------

y_preds = {
    'Random Forest': RF_pred, 
    'Logistic Regression': LR_pred,
    'Multilayer Perceptron': MLP_pred
}

model_metrics = []
for algogithm, y_pred in y_preds.items():
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0) #added because when no positive predictions these metrics are undefined 
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    
    model_metrics.append({'Classifier / Metric': algogithm, 'Precision': round(precision, 3), 'Recall': round(recall, 3), 'F1 Score': round(f1, 3), 'Accuracy': round(accuracy, 3)})


# Convert results to a DataFrame
results_df = pd.DataFrame(model_metrics)
print(results_df)

fig, ax = plt.subplots(figsize=(20, 4)) 
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')

table.auto_set_column_width(col=list(range(len(results_df.columns))))
table.auto_set_font_size(False)
table.set_fontsize(12)  # Set a readable font size

# Adjust borders and cell padding
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')  # Add border color
    cell.set_linewidth(1.5)  # Thicker border lines

table.scale(1.8, 2) 

plt.show()


# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Compute confusion matrix
# cm = confusion_matrix(y_test, RF_pred, labels=[1,2])

# # Display confusion matrix
# # 1 = illicit
# custom_labels = ["illicit:1", "licit:2"]
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_labels)
# disp.plot(cmap=plt.cm.Blues)

# # Show plot
# plt.show()



