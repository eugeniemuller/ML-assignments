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
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN

base = Path("/Users/eugeniemuller/HONOURS/HONOURS/ML/git assignments/ML-assignments/assignment3")
os.chdir(base)

#%%loading data 

coffee = pd.read_excel(base / "data/Coffee Shop Sales.xlsx")
coffee.to_csv(base / "data/coffee_shop_sales.csv", index=False)
coffee_data = pd.read_csv(base / "data/coffee_shop_sales.csv")

# %% data inspection

# store location and store id all the same information
# dropped most specific details for product category and product detail, as they are not relevant for the prediction of transaction quantity
coffee_data = coffee_data.drop(columns = ['transaction_id', 'store_location', 'product_category', 'product_detail'])

def convert_times():
    coffee_data['transaction_time'] = pd.to_datetime(coffee_data['transaction_time'], format='%H:%M:%S').dt.time

    # Define a function to categorize the time of day
    def categorize_time_of_day(time):
        if time < pd.to_datetime('12:00:00').time():
            return 'morning'
        elif time < pd.to_datetime('18:00:00').time():
            return 'afternoon'
        else:
            return 'evening'

    # Apply the function to create a new column
    coffee_data['time_of_day'] = coffee_data['transaction_time'].apply(categorize_time_of_day)
    coffee_data.drop(columns=['transaction_time'], inplace=True)


convert_times()
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

#%%
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

#%% scaling

# While a Robust Scaler prevents outliers from ruining the scaling of your other data, 
# it does not remove the outliers. 
# The extreme values will still exist in your scaled dataset—they will just 
# be pushed far outside the standard \([-1, 1]\) or \([0, 1]\) range. 
# If those outliers are errors (like sensor noise), you should still clip or 
# remove them. If they are real, legitimate spikes (like a sudden stock market crash), 
# a Robust Scaler is exactly what you need to keep your model stable

X = coffee_data.drop(columns=['transaction_qty'])
y = coffee_data['transaction_qty']

scaler = RobustScaler()
scaled_data = scaler.fit_transform(X['unit_price'].values.reshape(-1, 1))
X['unit_price'] = scaled_data

#%% encoding for the categorical feature
# using standard label encoding
le = LabelEncoder()
for col in X.select_dtypes(exclude=np.number).columns if col != 'transaction_date' else []:
    X[col] = le.fit_transform(X[col])


#%% functions : 
# create sliding window sequences : 

def create_sequences(X, y, window_size):
    sequences = []
    labels = []
    for i in range(len(X) - window_size):
        sequences.append(X.iloc[i:i + window_size].values)
        labels.append(y.iloc[i + window_size])
    return np.array(sequences), np.array(labels)


# getting correct data for models : 
def data_for_model(data, num_classes):
    transaction_ranking = data['transaction_qty'].value_counts().sort_values(ascending=True)
    least_represented_classes = transaction_ranking.index[:num_classes]

    data_for_model = data[data['transaction_qty'].isin(least_represented_classes)]
    return data_for_model

# train test split: 

def train_test(data):

    # data['transaction_date'] = pd.to_datetime(data['transaction_date'])

    # # Extract numerical components
    # data['month'] = data['transaction_date'].dt.month
    # data['day_of_week'] = data['transaction_date'].dt.dayofweek

    # # Drop the original date string column
    # data = data.drop('transaction_date')

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# neural network implementation

#%% RNN functions : 
def training_model(model, X_train, y_train, X_test, y_test):
    # Train the model

    history = model.fit(
        X_train, y_train, 
        epochs=20, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Generate predictions on test data
    predictions = model.predict(X_test)

    # Inverse transform predictions back to original scale for interpretation
    predictions_actual = predictions
    y_test_actual = y_test
    
    return history, predictions_actual, y_test_actual

#%% 
# first model 
# no hidden layers, only 2 least represented classes

X, y = create_sequences(X, y, window_size=5)

processed_data = pd.concat([X, y], axis=1)
data = data_for_model(processed_data, num_classes=2)
X_train, X_test, y_train, y_test = train_test(data)
LOOKBACK = 5  # Number of previous time steps to consider for prediction

#%%
# Initialize a sequential model
model_no_hidden = Sequential([
    # Input shape is (time_steps, features)
    SimpleRNN(1, input_shape=(LOOKBACK, X_train.shape[2]), activation='linear'),
    # Predicting a single continuous value
])

#%%
model_no_hidden.compile(optimizer='adam', loss='mse')
model_no_hidden.summary()
training_history_no_hidden, predictions_no_hidden, y_test_actual_no_hidden = training_model(model_no_hidden, X_train, y_train, X_test, y_test)

#%%



