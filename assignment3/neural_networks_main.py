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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN

tf.random.set_seed(42)
np.random.seed(42)

base = Path("/Users/eugeniemuller/HONOURS/HONOURS/ML/git assignments/ML-assignments/assignment3")
os.chdir(base)

#%%loading data ------------------------------------------------------------------------------------------------------------------------------

white = pd.read_csv(base / "data/winequality-white.csv", delimiter=';')
red = pd.read_csv(base / "data/winequality-red.csv", delimiter=';')
wine_data = pd.concat([white, red], ignore_index=True)
   
#%% data processing functions : ------------------------------------------------------------------------------------------------------------------------------
def robust_scaler(X_train, X_test):
    scaler = RobustScaler()
    X_train[:] = scaler.fit_transform(X_train)
    X_test[:] = scaler.transform(X_test)
    return X_train, X_test

# data inspection : -------------------------------------------------------------------------------------------------------------------------

#%% missingness : -------------------------------------------------------------------------------------------------------------------------------
shape = wine_data.shape
print("Shape of the dataset:", shape)
missingness = wine_data.isnull().sum()/len(wine_data)*100
print("Percentage Missingness in the dataset:\n", missingness)

#%% data exploration : ---------------------------------------------------------------------------------------------------------------------------
classes = wine_data['quality'].value_counts().sort_index()
print("Class distribution:\n", classes)

sns.countplot(data=wine_data, x='quality', palette='Set2')
plt.title('Distribution of Target Classes')
plt.xlabel('Class Label (Transaction Quantity)')
plt.ylabel('Count')
plt.show()
plt.savefig(base / "findings/class_distribution.png")
plt.close()

# for col in wine_data.columns:
#     plt.figure(figsize=(8, 6))
#     sns.countplot(data=wine_data, x=col, palette='Set2')
#     plt.title(f'Distribution of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Count')
#     plt.show()
#     plt.close()

#%% skewness : ---------------------------------------------------------------------------------------------------------------------------------------
skew_table = pd.DataFrame(columns=['Feature', 'Skewness'])
for col in wine_data.columns:
    skewness = calculate_skew(wine_data[col], bias = False)
    skew_table = pd.concat([skew_table, pd.DataFrame({'Feature': [col], 'Skewness': [skewness]})], ignore_index=True)

skew_table.to_csv(base / "findings/skewness_table.csv", index=False)
print(skew_table)


#%% outliers : ---------------------------------------------------------------------------------------------------------------------------------------
outlier_table = pd.DataFrame(columns=['Feature', 'Outlier Count'])
for col in wine_data.columns:
    Q1 = wine_data[col].quantile(0.25)
    Q3 = wine_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = ((wine_data[col] < lower_bound) | (wine_data[col] > upper_bound)).sum()
    outlier_table = pd.concat([outlier_table, pd.DataFrame({'Feature': [col], 'Outlier Count': [outlier_count]})], ignore_index=True)

outlier_table.to_csv(base / "findings/outlier_table.csv", index=False)
print(outlier_table)

# many outliers, will need to do clipping or robust scaling. Will use robust scaling for this assignment.

#%% modeling functions : ---------------------------------------------------------------------------------------------------------------------------------

# train, test splits
def train_test(data):
    X = data.iloc[:, :-1]
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X_train.copy()
    X_test = X_test.copy()
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
    print("Getting data for stage", stage, "with classes:", classes)
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

# have 4 stages 
# first stage : 2 least represented classes 
# second stage : 2 next least represented classes 
# third stage : 2 next least represented classes
# fourth stage : most represented class 
STAGES = [0, 2, 2, 2, 1]  # number of classes added introduced at each stage

#%% model data initialization : --------------------------------------------------------------------------------------------------------------------------------------------

class_order = get_class_order(wine_data['quality'])
X_train, X_test, y_train, y_test = train_test(wine_data)
scaled_X_train, scaled_X_test = robust_scaler(X_train, X_test)

n_stages = len(class_order) -  1            

print("Processing stage", STAGES[0])
X_s, y_s, classes = data_for_stage(X_train, y_train, STAGES[0], class_order, n_initial=2, cumulative=True)
y_s = y_s.astype('float32')  

X_t, y_t, _ = data_for_stage(X_test, y_test, STAGES[0], class_order, n_initial=2, cumulative=True)
y_t = y_t.astype('float32')

#%% first model : ---------------------------------------------------------------------------------------------------------------------------------------------------
# no hidden layers, only 2 least represented classes

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Define a model with NO hidden layers
model1 = Sequential([
    # The input shape is specified, and it maps directly to the output layer
    layers.Dense(units=10, input_shape=(11,), activation='sigmoid')
])

model1.compile(optimizer="adam",
              loss=loss, metrics=['sparse_categorical_accuracy'])
model1.summary()

history = model1.fit(X_s, y_s, epochs=10, batch_size=32, validation_split=0.2)
get_loss(history)

#%%
predictions = model1.predict(X_t)
comparison = pd.DataFrame({'Actual': y_t, 'Predicted': np.argmax(predictions, axis=1)})
print(comparison)



# %%
