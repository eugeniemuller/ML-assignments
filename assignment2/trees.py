
# CLASSIFICATION TREES : 

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

#%%
traffic_data = pd.read_csv("networktraffic.csv", delimiter=",")

traffic_data = traffic_data.drop_duplicates()

#%%
# missingness ------------------------------------------------------------------------------------------------------
missingness = []
for col in traffic_data.columns :
    na = (traffic_data[col] == "?").sum()
    if na > 0.20 * len(traffic_data):
        missingness.append(col)

traffic_data[missingness] = traffic_data[missingness].replace({'?' : 'missing'})

# splits ------------------------------------------------------------------------------------------------------
#%%
X = traffic_data.iloc[:, :-1]
y = traffic_data["attack_cat"]

numerical = X.select_dtypes(include=np.number).columns
categorical = X.select_dtypes(exclude=np.number).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train = X_train.copy()
X_test = X_test.copy()

#%%
# cardinality = nunique / length -------------------------------------------------------------------------------------
card_columns = []

for col in X_train.columns:
    ratio = X_train[col].nunique() / len(X_train)
    if ratio == 1 :
        card_columns.append(col)

X_train_card = X_train.drop(columns = card_columns)
X_test_card = X_test.drop(columns = card_columns)

#%%
# redundant / correlation ------------------------------------------------------------------------------------------
# irrelevant /redundant features (feature selection / dim reduction)
corr = X_train_card.corr(numeric_only=True).abs()
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
threshold = 0.85
pairs = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
X_train_corr = X_train_card.drop(columns = pairs)
X_test_corr = X_test_card.drop(columns = pairs)

# missingness handling stays the same (the "?" -> "missing" category trick)
# duplicates stays the same
# split stays the same

# no PowerTransformer, no RobustScaler needed
# no missingness

#%%
# smotenc --------------------------------------------------------------------------------------------------------------
# recompute which categorical columns actually survived filtering
categorical_remaining = [col for col in categorical if col in X_train_corr.columns]

# convert to integer positions within X_train_corr
cat_positions = [X_train_corr.columns.get_loc(col) for col in categorical_remaining]

smote_nc = SMOTENC(categorical_features=cat_positions, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X_train_corr, y_train)

#%%
# ordinal encode categoricals instead of one-hot ------------------------------------------------------------------------
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_cat = pd.DataFrame(ord_enc.fit_transform(X_resampled[categorical]), columns=categorical, index=X_resampled.index)
X_test_cat = pd.DataFrame(ord_enc.transform(X_test_corr[categorical]), columns=categorical, index=X_test_corr.index)

#%%
X_train_final = pd.concat([X_resampled.select_dtypes(include = np.number), X_train_cat], axis=1)
X_test_final = pd.concat([X_test_corr.select_dtypes(include = np.number), X_test_cat], axis=1)


#%%
tree = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_leaf': [2, 4, 6],
    'min_samples_split': [4, 8, 12],
    'max_features': [None, 'sqrt', 'log2']
}

# 4. Set up Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# 5. Execute hyperparameter tuning
print("Starting grid search exploration...")
grid_search.fit(X_train_final, y_resampled)

# 6. Extract and evaluate the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\n=== Tuning Results ===")
print(f"Best Hyperparameters: {best_params}")

#%%
tree.fit(X_train_final, y_resampled)
predictions = tree.predict(X_test_final)

accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
# %%
