
#%%
import numpy as np
import pandas as pd
from scipy.stats import skew as calculate_skew
from scipy.stats import chi2_contingency
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.covariance import LedoitWolf


#%%
traffic_data = pd.read_csv("networktraffic.csv", delimiter=",")

# KNN : ----------------------------------------------------------------------------------------

# missingness ---------------------------------------------------------------------------------------------------
# missing values needs to be fixed : drop rows / impute values 

# %%

missingness = []
for col in traffic_data.columns :
    na = (traffic_data[col] == "?").sum()
    if na > 0.20 * len(traffic_data):
        missingness.append(col)

# crosstab = pd.crosstab(traffic_data[missingness], traffic_data["attack_cat"])
# after looking at crosstab - looks like MAR

is_missing_service = []
for i in traffic_data[missingness] :  
    is_missing_service.append('1') if i == '?' else is_missing_service.append('0')

service_target_ctable = pd.crosstab(is_missing_service, traffic_data["attack_cat"])
chi2, p, dof, expected = chi2_contingency(service_target_ctable)
service_target_ctable.to_csv("service_table.csv", index=True)

# p-value = 0
# missing at random
# missingnes is statistically significant - make missingness category

traffic_data[missingness] = traffic_data[missingness].replace({'?' : 'missing'})

# duplicates -----------------------------------------------------------------------------------------
#%%
traffic_data = traffic_data.drop_duplicates()

# train and test splits -----------------------------------------------------------------------------------------
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
# numerical skewness ---------------------------------------------------------------------------------------------
# for col in numerical :
#     skewness = calculate_skew(X[col], bias = False)
#     print(f"{col} skewness: {skewness}")


pt = PowerTransformer(method='yeo-johnson')
X_train_skew_transformed = pt.fit_transform(X_train[numerical])
X_test_skew_transformed = pt.transform(X_test[numerical])

X_train_num = pd.DataFrame(X_train_skew_transformed, columns=numerical, index=X_train.index)
X_test_num = pd.DataFrame(X_test_skew_transformed, columns=numerical, index=X_test.index)


#%%
# stdise ---------------------------------------------------------------------------------------------------------
# feature scaling (since working with distances) - need to stdize

scalar = RobustScaler()
X_train_stdised = scalar.fit_transform(X_train_num[numerical])
X_test_stdised = scalar.transform(X_test_num[numerical])

X_train_num = pd.DataFrame(X_train_stdised, columns=numerical, index=X_train.index)
X_test_num = pd.DataFrame(X_test_stdised, columns=numerical, index=X_test.index)

#%%
# outliers :  -----------------------------------------------------------------------------------------------------

# looking for EXTREME outliers

q1 = X_train_num[numerical].quantile(0.25)
q3 = X_train_num[numerical].quantile(0.75)
iqr = q3 - q1

# FOR REPORT WRITING : 

outliers_iqr = (X_train_num[numerical] < (q1 - 3 * iqr)) | (X_train_num[numerical] > (q3 + 3 * iqr))
out_pct = (outliers_iqr.sum()/len(traffic_data[numerical]))*100

outlier_flag = outliers_iqr.any(axis=1) 
crosstab_outlier = pd.crosstab(outlier_flag, y)
crosstab_outlier.to_csv("outlier crosstab", index = True)

# # to show why we need to use mahalanobis distance : 

outlier_by_feature = outliers_iqr.groupby(y).mean()  # % outlier rate per feature per class
print(outlier_by_feature)

lower = q1 - 3 * iqr
upper = q3 + 3 * iqr

X_train_clipped = pd.concat(
    [X_train_num[numerical].clip(lower=lower, upper=upper, axis=1), X_train[categorical]],
    axis=1
)
X_test_clipped = pd.concat(
    [X_test_num[numerical].clip(lower=lower, upper=upper, axis=1), X_test[categorical]],
    axis=1
)

# each category's "extremeness" lives in a different, largely non-overlapping subset of features
# Mahalanobis accounts for the covariance structure, effectively "listening" more carefully to the 
# directions where variance concentrates — which is exactly what you want when different classes are 
# extreme in different feature subspaces

#%%
# cardinality = nunique / length -------------------------------------------------------------------------------------
card_columns = []

for col in X_train_clipped.columns:
    ratio = X_train_clipped[col].nunique() / len(X_train_clipped)
    if ratio == 1 :
        card_columns.append(col)

X_train_card = X_train_clipped.drop(columns = card_columns)
X_test_card = X_test_clipped.drop(columns = card_columns)

#%%
# redundant / correlation ------------------------------------------------------------------------------------------
# irrelevant /redundant features (feature selection / dim reduction)
corr = X_train_card.corr(numeric_only=True).abs()
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
threshold = 0.85
pairs = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
X_train_red = X_train_card.drop(columns = pairs)
X_test_red = X_test_card.drop(columns = pairs)

# one-hot encoding ----------------------------------------------------------------------------------------------------
# categorical variables - one hot 
#%%
X_train_final = pd.get_dummies(X_train_red, columns=categorical, dtype=int)
X_test_final = pd.get_dummies(X_test_red, columns=categorical, dtype=int)
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

 
#%%
# grid search under the Mahalanobis distance measure -----------------------------------------------------------------
# this replaces the previous hardcoded n_neighbors=3: the number of neighbours
# is now tuned under the same distance measure the final model actually uses

SUBSAMPLE_SIZE = 20000
X_sub, _, y_sub, _ = train_test_split(
    X_train_final, y_train, train_size=SUBSAMPLE_SIZE, stratify=y_train, random_state=42
)

cov_sub = LedoitWolf().fit(X_sub)
 
mahalanobis_param_grid = {'n_neighbors': np.arange(1, 4)}
knn_mahalanobis_sub = KNeighborsClassifier(
    weights='distance',
    metric='mahalanobis',
    metric_params={'VI': cov_sub.precision_},
    algorithm='auto'
)
knn_maha_cv = GridSearchCV(knn_mahalanobis_sub, mahalanobis_param_grid, cv=5, n_jobs=-1, verbose=2)
knn_maha_cv.fit(X_sub, y_sub)
 
best_idx = knn_maha_cv.best_index_
cv_mean = knn_maha_cv.cv_results_['mean_test_score'][best_idx]
cv_std = knn_maha_cv.cv_results_['std_test_score'][best_idx]
best_k = knn_maha_cv.best_params_['n_neighbors']
 
print(f"\n=== Mahalanobis-metric tuning (subsample, n={SUBSAMPLE_SIZE}) ===")
print(f"Best Hyperparameters: {knn_maha_cv.best_params_}")
print(f"CV Accuracy (on subsample): {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
 
#%%

from scipy.linalg import cholesky

cov = LedoitWolf().fit(X_train_final)
L = cholesky(cov.precision_, lower=True)
 
X_train_whitened = X_train_final.to_numpy() @ L
X_test_whitened = X_test_final.to_numpy() @ L
 
best_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights='distance',
    metric='euclidean',
    algorithm='auto'
)
best_model.fit(X_train_whitened, y_train)
 
#%%
# train vs test accuracy -----------------------------------------------------------------------------------------

# predict on the whitened arrays, matching what best_model was fit on -

train_predictions = best_model.predict(X_train_whitened)
test_predictions = best_model.predict(X_test_whitened)
 
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
 
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Train-Test Gap: {(train_accuracy - test_accuracy) * 100:.2f}%")
 
# %%
comparison = pd.DataFrame({
    "true y" : y_test,
    "predicted y" : test_predictions
})

 
# %%
comparison = pd.DataFrame({
    "true y" : y_test,
    "predicted y" : test_predictions
})
 
#%%
# repeated runs for mean/std test accuracy -----------------------------------------------------------------------
# hyperparameters are fixed from the Mahalanobis grid search above; only the


N_REPEATS = 10
repeat_accuracies = []
 
for seed in range(N_REPEATS):
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    Xr_train = Xr_train.copy()
    Xr_test = Xr_test.copy()
 
    pt_r = PowerTransformer(method='yeo-johnson')
    Xr_train_num = pd.DataFrame(
        pt_r.fit_transform(Xr_train[numerical]), columns=numerical, index=Xr_train.index
    )
    Xr_test_num = pd.DataFrame(
        pt_r.transform(Xr_test[numerical]), columns=numerical, index=Xr_test.index
    )
 
    scalar_r = RobustScaler()
    Xr_train_num = pd.DataFrame(
        scalar_r.fit_transform(Xr_train_num[numerical]), columns=numerical, index=Xr_train.index
    )
    Xr_test_num = pd.DataFrame(
        scalar_r.transform(Xr_test_num[numerical]), columns=numerical, index=Xr_test.index
    )
 
    q1_r = Xr_train_num[numerical].quantile(0.25)
    q3_r = Xr_train_num[numerical].quantile(0.75)
    iqr_r = q3_r - q1_r
    lower_r = q1_r - 3 * iqr_r
    upper_r = q3_r + 3 * iqr_r
 
    Xr_train_clipped = pd.concat(
        [Xr_train_num[numerical].clip(lower=lower_r, upper=upper_r, axis=1), Xr_train[categorical]], axis=1
    )
    Xr_test_clipped = pd.concat(
        [Xr_test_num[numerical].clip(lower=lower_r, upper=upper_r, axis=1), Xr_test[categorical]], axis=1
    )
 
    Xr_train_card = Xr_train_clipped.drop(columns=card_columns)
    Xr_test_card = Xr_test_clipped.drop(columns=card_columns)
 
    Xr_train_red = Xr_train_card.drop(columns=pairs)
    Xr_test_red = Xr_test_card.drop(columns=pairs)
 
    Xr_train_final = pd.get_dummies(Xr_train_red, columns=categorical, dtype=int)
    Xr_test_final = pd.get_dummies(Xr_test_red, columns=categorical, dtype=int)
    Xr_test_final = Xr_test_final.reindex(columns=Xr_train_final.columns, fill_value=0)
 
    cov_r = LedoitWolf().fit(Xr_train_final)
    model_r = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric='mahalanobis',
        metric_params={'VI': cov_r.precision_},
        algorithm='auto'
    )
    model_r.fit(Xr_train_final, yr_train)
    preds_r = model_r.predict(Xr_test_final)
    repeat_accuracies.append(accuracy_score(yr_test, preds_r))
 
repeat_accuracies = np.array(repeat_accuracies)
print(f"\n=== Repeated Runs (n={N_REPEATS}) ===")
print(f"Test Accuracy: {repeat_accuracies.mean() * 100:.2f}% (+/- {repeat_accuracies.std() * 100:.2f}%)")
# %%