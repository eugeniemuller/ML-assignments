#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew as calculate_skew
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

#%%loading data 
base = Path("/Users/eugeniemuller/HONOURS/HONOURS/ML/git assignments/ML-assignments/assignment3")
os.chdir(base)

coffee = pd.read_excel(base / "data/Coffee Shop Sales.xlsx")
coffee.to_csv(base / "data/coffee_shop_sales.csv", index=False)
coffee_data = pd.read_csv(base / "data/coffee_shop_sales.csv")

# %% data inspection

# store location and store id all the same information
# dropped most specific details for product category and product detail, as they are not relevant for the prediction of transaction quantity
coffee_data = coffee_data.drop(columns = ['transaction_id', 'store_location', 'product_category', 'product_detail'])
numerical = coffee_data.select_dtypes(include=[np.number])
categorical = coffee_data.select_dtypes(exclude=[np.number])

#%% missingness
shape = coffee_data.shape
print("Shape of the dataset:", shape)
missingness = coffee_data.isnull().sum()/len(coffee_data)*100
print("Percentage Missingness in the dataset:\n", missingness)

#%% data exploration
classes = coffee_data['transaction_qty'].value_counts().sort_index()
print("Class distribution:\n", classes)

sns.countplot(data=coffee_data, x='transaction_qty', palette='Set2')
plt.title('Distribution of Target Classes')
plt.xlabel('Class Label (Transaction Quantity)')
plt.ylabel('Count')
plt.show()
plt.savefig(base / "findings/class_distribution.png")
plt.close()

#%%
# skewness
skew_table = pd.DataFrame(columns=['Feature', 'Skewness'])
for col in numerical.columns:
    skewness = calculate_skew(numerical[col], bias = False)
    skew_table = pd.concat([skew_table, pd.DataFrame({'Feature': [col], 'Skewness': [skewness]})], ignore_index=True)

skew_table.to_csv(base / "findings/skewness_table.csv", index=False)
print(skew_table)


for col in categorical.columns:
    if col != 'transaction_date' and col != 'transaction_time':
        plt.figure(figsize=(8, 6))
        sns.countplot(data=coffee_data, x=col, palette='Set2')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()
        plt.close()

#%%
#outliers
outlier_table = pd.DataFrame(columns=['Feature', 'Outlier Count'])
for col in numerical.columns:
    Q1 = numerical[col].quantile(0.25)
    Q3 = numerical[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = ((numerical[col] < lower_bound) | (numerical[col] > upper_bound)).sum()
    outlier_table = pd.concat([outlier_table, pd.DataFrame({'Feature': [col], 'Outlier Count': [outlier_count]})], ignore_index=True)

outlier_table.to_csv(base / "findings/outlier_table.csv", index=False)
print(outlier_table)

# %% scaling : 

X_data = coffee_data.iloc[:, :-1]
y = coffee_data['transaction_qty']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, stratify=y, random_state=42
)

X_train = X_train.copy()
X_test = X_test.copy()

#%%

# While a Robust Scaler prevents outliers from ruining the scaling of your other data, 
# it does not remove the outliers. 
# The extreme values will still exist in your scaled dataset—they will just 
# be pushed far outside the standard \([-1, 1]\) or \([0, 1]\) range. 
# If those outliers are errors (like sensor noise), you should still clip or 
# remove them. If they are real, legitimate spikes (like a sudden stock market crash), 
# a Robust Scaler is exactly what you need to keep your model stable

scalar = RobustScaler()
X_train_stdised = scalar.fit_transform(X_train[numerical.columns])
X_test_stdised = scalar.transform(X_test[numerical.columns])

X_train_num = pd.DataFrame(X_train_stdised, columns=numerical.columns, index=X_train.index)
X_test_num = pd.DataFrame(X_test_stdised, columns=numerical.columns, index=X_test.index)

# %%
