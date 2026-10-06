#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb

import os

main = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(main, 'data')

# data from given path
traffic_data = pd.read_csv(os.path.join(data_path, 'networkTraffic.csv'), delimiter=",")

# data overview
numerical = traffic_data.select_dtypes(include=np.number)
categorical = traffic_data.select_dtypes(exclude=np.number)

#%%
# data preprocessing

# missingness : ------------------------------------------------------------------------------------------
missingness = []
for col in traffic_data.columns :
    na = (traffic_data[col] == "?").sum()
    if na > 0.20 * len(traffic_data):
        missingness.append(col)

traffic_data[missingness] = traffic_data[missingness].replace({'?' : 'missing'})

# duplicates ------------------------------------------------------------------------------------------------------
traffic_data = traffic_data.drop(columns="id")
n_before = len(traffic_data)
traffic_data = traffic_data.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_before - len(traffic_data)} duplicate rows")


#%% 
#train-test splits

X = traffic_data.iloc[:, :-1]
y = traffic_data["attack_cat"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()  

#%% 
# catboost implementation 


categorical_seq = [c for c in categorical.columns if c in X_train.columns]

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.5,
    depth=8,
    eval_metric="TotalF1:average=Macro",
    early_stopping_rounds=50,     # stop when val macro-F1 stops improving
    use_best_model=True,
    random_seed=42,
    verbose=50,
)
cat_model.fit(X_train, y_train, cat_features=categorical_seq,
              eval_set=(X_val, y_val))

preds = cat_model.predict(X_test).ravel()
print("Best iteration:", cat_model.get_best_iteration())
print("Test macro-F1:     ", f1_score(y_test, preds, average="macro"))
print("Test balanced acc: ", balanced_accuracy_score(y_test, preds))
print("Test accuracy:     ", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, digits=3))
# %%
