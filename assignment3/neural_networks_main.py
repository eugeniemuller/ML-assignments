#%%
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew as calculate_skew
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN


tf.random.set_seed(42)
np.random.seed(42)

base = Path("/Users/eugeniemuller/HONOURS/HONOURS/ML/git assignments/ML-assignments/assignment3")
os.chdir(base)

#%%loading data ------------------------------------------------------------------------------------------------------------------------------

coffee = pd.read_excel(base / "data/Coffee Shop Sales.xlsx")
coffee.to_csv(base / "data/coffee_shop_sales.csv", index=False)
coffee_data = pd.read_csv(base / "data/coffee_shop_sales.csv")

#%% pre processing variables : ----------------------------------------------------------------------------------------------------------------
WINDOW = 5         

#%% pre processing functions : ----------------------------------------------------------------------------------------------------------------

def convert_times(coffee_data):
    coffee_data["timestamp"] = pd.to_datetime(coffee_data["transaction_date"] + ' ' + coffee_data["transaction_time"]) 
    coffee_data = coffee_data.sort_values(by="timestamp").reset_index(drop=True)
    hour = coffee_data['timestamp'].dt.hour
    coffee_data["time_of_day"] = np.select([hour < 12, hour < 18], ["morning", "afternoon"], "evening")
    coffee_data["day_of_week"] = coffee_data["timestamp"].dt.dayofweek
    coffee_data["month"] = coffee_data["timestamp"].dt.month
    coffee_data.drop(columns=['transaction_time', 'transaction_date', 'timestamp'], inplace=True)
    return coffee_data

def robust_scaler(X_train, X_test):
    scaler = RobustScaler()
    X_train[:] = scaler.fit_transform(X_train)
    X_test[:] = scaler.transform(X_test)
    return X_train, X_test

# %% data inspection : -------------------------------------------------------------------------------------------------------------------------

# store location and store id all the same information
# dropped most specific details for product category and product detail, as they are not relevant for the prediction of transaction quantity
coffee_data = coffee_data.drop(columns = ['transaction_id', 'store_location', 'product_category', 'product_detail'])
coffee_data = convert_times(coffee_data)
numerical = coffee_data.select_dtypes(include=[np.number])
categorical = coffee_data.select_dtypes(exclude=[np.number])

#%% missingness : -------------------------------------------------------------------------------------------------------------------------------
shape = coffee_data.shape
print("Shape of the dataset:", shape)
missingness = coffee_data.isnull().sum()/len(coffee_data)*100
print("Percentage Missingness in the dataset:\n", missingness)

#%% data exploration : ---------------------------------------------------------------------------------------------------------------------------
classes = coffee_data['transaction_qty'].value_counts().sort_index()
print("Class distribution:\n", classes)

sns.countplot(data=coffee_data, x='transaction_qty', palette='Set2')
plt.title('Distribution of Target Classes')
plt.xlabel('Class Label (Transaction Quantity)')
plt.ylabel('Count')
plt.show()
plt.savefig(base / "findings/class_distribution.png")
plt.close()

for col in categorical.columns:
    if col != 'transaction_date' and col != 'transaction_time':
        plt.figure(figsize=(8, 6))
        sns.countplot(data=coffee_data, x=col, palette='Set2')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()
        plt.close()

#%% skewness : ---------------------------------------------------------------------------------------------------------------------------------------
skew_table = pd.DataFrame(columns=['Feature', 'Skewness'])
for col in numerical.columns:
    skewness = calculate_skew(numerical[col], bias = False)
    skew_table = pd.concat([skew_table, pd.DataFrame({'Feature': [col], 'Skewness': [skewness]})], ignore_index=True)

skew_table.to_csv(base / "findings/skewness_table.csv", index=False)
print(skew_table)


#%% outliers : ---------------------------------------------------------------------------------------------------------------------------------------
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

#%% encoding for the categorical feature : ---------------------------------------------------------------------------------------------------------------
# using standard label encoding

le = LabelEncoder()
for col in categorical.columns:
    if col != 'transaction_date':
        coffee_data[col] = le.fit_transform(coffee_data[col])

#%% modeling functions : ---------------------------------------------------------------------------------------------------------------------------------

# sliding windows
def create_sequences(X, y, window):
    X = X.to_numpy(dtype="float32")
    # last row of each window = the row being predicted
    idx = np.arange(window - 1, len(X))          
    seqs = np.stack([X[i - window + 1:i + 1] for i in idx])
    labels = y[idx]
    return seqs, labels

# train, test splits
def train_test(X_all, y_all):

    train_size = int(len(X_all) * 0.8)
    X_all = X_all.astype(np.float32)
    X_train, X_test = X_all.iloc[:train_size].copy(), X_all.iloc[train_size:].copy()
    y_train, y_test = y_all[:train_size], y_all[train_size:]
    return X_train, X_test, y_train, y_test

def get_class_order(y):
    labels, counts = np.unique(y, return_counts=True)
    class_order = labels[np.argsort(counts)].tolist()
    return class_order

def data_for_stage(X, y, stage, class_order, n_initial, cumulative=True):
    # Get the classes for the current stage

    n_seen = n_initial + stage
    if n_seen > len(class_order):
        raise ValueError(f"stage {stage} is out of range: only {len(class_order) - n_initial} incremental stages exist")

    if cumulative:
        classes = class_order[:n_seen]
    elif stage == 0:
        classes = class_order[:n_initial]
    else:
        classes = [class_order[n_seen - 1]]

    mask = np.isin(y, classes)
    return X[mask], y[mask], classes

def get_loss(history):
    final_loss = history.history['loss'][-1]
    print(f"Final Training Loss (MSE): {final_loss:.4f}")


# neural network implementation


#%% model variables : --------------------------------------------------------------------------------------------------------------------------------------------

EPOCHS = 3            # per incremental stage
BATCH = 256
REPLAY_PER_CLASS = 500  # exemplars kept per old class (0 = plain fine-tuning)
 
# Class-incremental schedule: which transaction_qty values are introduced at each stage.
# (Classes 4, 6, 8 have only 23, 3 and 10 rows in total, so they are grouped into one stage.)

# The stages are defined as lists of class indices.
# The first stage includes classes 1 and 2.
# The second stage includes class 3.
# The third stage includes classes 4, 6, and 8.

LOOKBACK = 5  # number of previous time steps to use as input features
STAGES = [[1, 2], [3], [4, 6, 8]]

#%% model data initialization : --------------------------------------------------------------------------------------------------------------------------------------------

transaction_qties = sorted(coffee_data['transaction_qty'].unique())
qties_to_class = {qty: idx for idx, qty in enumerate(transaction_qties)}
class_to_qties = {idx: qty for qty, idx in qties_to_class.items()}
K = len(transaction_qties)
y_all = coffee_data["transaction_qty"].map(qties_to_class).to_numpy()
X_all = coffee_data.drop(columns=['transaction_qty'])
X_all["prev_qty"] = pd.Series(y_all).shift(1).fillna(0).to_numpy()

X_train_df, X_test_df, y_train_raw, y_test_raw = train_test(X_all, y_all)
X_train, X_test = robust_scaler(X_train_df, X_test_df)

X_train, y_train = create_sequences(X_train_df, y_train_raw, WINDOW)
X_test, y_test = create_sequences(X_test_df, y_test_raw, WINDOW)
n_features = X_train.shape[2]
print("train:", X_train.shape, "test:", X_test.shape)

class_order = get_class_order(y_train)         
print([class_to_qties[c] for c in class_order])  

n_stages = len(class_order) - 2 + 1            
for stage in range(n_stages):
    X_s, y_s, classes = data_for_stage(X_train, y_train, stage, class_order, n_initial=2, cumulative=True)
    print(stage, [class_to_qties[c] for c in classes], X_s.shape)

#%% first model : ---------------------------------------------------------------------------------------------------------------------------------------------------

# no hidden layers, only 2 least represented classes

# Initialize a sequential model
model_no_hidden = Sequential([
    # Input shape is (time_steps, features)
    SimpleRNN(1, input_shape=(LOOKBACK, X_train.shape[2]), activation='linear'),
    # Predicting a single continuous value
])

#%% fitting and training the model : ---------------------------------------------------------------------------------------------------------------------------------------------------
model_no_hidden.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model_no_hidden.summary()

history = model_no_hidden.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
get_loss(history)

#%% testing the model : ---------------------------------------------------------------------------------------------------------------------------------------------------


# Test data: You can reuse data_for_stage(X_test, y_test, stage, class_order) 
# for evaluation, so the test set grows with the classes the model has seen.
# %%
