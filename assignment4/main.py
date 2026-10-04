#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool

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

# cardinality = nunique / length -------------------------------------------------------------------------------------
card_columns = []

for col in traffic_data.columns:
    ratio = traffic_data[col].nunique() / len(traffic_data)
    if ratio >= 0.85 :
        card_columns.append(col)

traffic_data.drop(card_columns, axis=1, inplace=True)

#%% 
#train-test splits

X = traffic_data.iloc[:, :-1]
y = traffic_data["attack_cat"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = X_train.copy()
X_test = X_test.copy()

#%% 
# catboost implementation 

model = CatBoostClassifier(
    iterations=10,        # Number of boosting rounds
    learning_rate=0.1,    # Step size shrinkage
    depth=4,              # Depth of the tree
    verbose=True          # Set to False or use silent=True to suppress logs
)

# 4. Fit the model
categorical_seq = [col for col in categorical.columns if col in X_train.columns]
model.fit(X_train, y_train, cat_features=categorical_seq, eval_set=(X_test, y_test), verbose=True)

# 5. Predict and Evaluate
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, preds)}")

# catboost
# cgboost
# lightgbm
# %%
