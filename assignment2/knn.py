
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
from sklearn.metrics import accuracy_score


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
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(1,4)}
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X_train_final,y_train)

print(knn_cv.best_params_)
print(knn_cv.best_score_)

#%%
from sklearn.covariance import LedoitWolf
cov = LedoitWolf().fit(X_train_final)
knn = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',
    metric='mahalanobis',
    metric_params={'VI': cov.precision_},
    algorithm='brute'
)

#%%
knn.fit(X_train_final, y_train)
predictions = knn.predict(X_test_final)

# %%
comparison = pd.DataFrame({
    "true y" : y_test,
    "predicted y" : predictions
})


