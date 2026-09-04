
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
X_train_cat = pd.DataFrame(ord_enc.fit_transform(X_resampled[categorical_remaining]), columns=categorical_remaining, index=X_resampled.index)
X_test_cat = pd.DataFrame(ord_enc.transform(X_test_corr[categorical_remaining]), columns=categorical_remaining, index=X_test_corr.index)

#%%
X_train_final = pd.concat([X_resampled.select_dtypes(include = np.number), X_train_cat], axis=1)
X_test_final = pd.concat([X_test_corr.select_dtypes(include = np.number), X_test_cat], axis=1)


#%%
tree = DecisionTreeClassifier(max_depth=10, random_state=42)

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

# mean and std of CV accuracy across the 5 folds, for the winning hyperparameter
# combination specifically (not just the single best_score_ scalar)
best_idx = grid_search.best_index_
cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
cv_std = grid_search.cv_results_['std_test_score'][best_idx]
 
print("\n=== Tuning Results ===")
print(f"Best Hyperparameters: {best_params}")
print(f"CV Accuracy: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")

#%%
# use the tuned estimator from the grid search, not the untuned "tree" object.
# grid_search(refit=True) already fits best_model on X_train_final/y_resampled,
# so it does not need to be fitted again here.
train_predictions = best_model.predict(X_train_final)
test_predictions = best_model.predict(X_test_final)
 
train_accuracy = accuracy_score(y_resampled, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
 
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Train-Test Gap: {(train_accuracy - test_accuracy) * 100:.2f}%")


#%%
# repeated runs for mean/std test accuracy -----------------------------------------------------
# hyperparameters are fixed from the grid search above; only the random seed governing
# the train/test split and the SMOTENC oversampling is varied across repetitions
N_REPEATS = 10
seeds = range(N_REPEATS)
repeat_accuracies = []
 
for seed in seeds:
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    Xr_train = Xr_train.copy()
    Xr_test = Xr_test.copy()
 
    Xr_train_card = Xr_train.drop(columns=card_columns)
    Xr_test_card = Xr_test.drop(columns=card_columns)
 
    Xr_train_corr = Xr_train_card.drop(columns=pairs)
    Xr_test_corr = Xr_test_card.drop(columns=pairs)
 
    smote_nc_r = SMOTENC(categorical_features=cat_positions, random_state=seed)
    Xr_resampled, yr_resampled = smote_nc_r.fit_resample(Xr_train_corr, yr_train)
 
    ord_enc_r = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    Xr_train_cat = pd.DataFrame(
        ord_enc_r.fit_transform(Xr_resampled[categorical_remaining]),
        columns=categorical_remaining, index=Xr_resampled.index
    )
    Xr_test_cat = pd.DataFrame(
        ord_enc_r.transform(Xr_test_corr[categorical_remaining]),
        columns=categorical_remaining, index=Xr_test_corr.index
    )
 
    Xr_train_final = pd.concat([Xr_resampled.select_dtypes(include=np.number), Xr_train_cat], axis=1)
    Xr_test_final = pd.concat([Xr_test_corr.select_dtypes(include=np.number), Xr_test_cat], axis=1)
 
    model_r = DecisionTreeClassifier(**best_params, random_state=seed)
    model_r.fit(Xr_train_final, yr_resampled)
    preds_r = model_r.predict(Xr_test_final)
    repeat_accuracies.append(accuracy_score(yr_test, preds_r))
 
repeat_accuracies = np.array(repeat_accuracies)
print(f"\n=== Repeated Runs (n={N_REPEATS}) ===")
print(f"Test Accuracy: {repeat_accuracies.mean() * 100:.2f}% (+/- {repeat_accuracies.std() * 100:.2f}%)")
# %%
 
