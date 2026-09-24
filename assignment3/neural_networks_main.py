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
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

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

for col in wine_data.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=wine_data, x=col, palette='Set2')
    plt.title(f'Distribution of {col}')
    plt.ylabel('Count')
    plt.show()
    plt.savefig(base / f"findings/class_distribution_{col}.png")
    plt.close()

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

#%% model variables : --------------------------------------------------------------------------------------------------------------------------------------------

EPOCHS = 50
BATCH = 256
REPLAY_PER_CLASS = 500  # exemplars kept per old class (0 = plain fine-tuning)
 
# Class-incremental schedule: which transaction_qty values are introduced at each stage.
# (Classes 4, 6, 8 have only 23, 3 and 10 rows in total, so they are grouped into one stage.)

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

def remap_labels(y, seen_classes):
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    return np.array([class_to_idx[v] for v in y])

def build_model(n_hidden_units, n_output_classes, n_features):
    layers_list = [layers.Input(shape=(n_features,))]
    if n_hidden_units > 0:
        layers_list.append(layers.Dense(n_hidden_units, activation='relu'))
    layers_list.append(layers.Dense(n_output_classes, activation='softmax'))
    model = Sequential(layers_list)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])
    return model

def diagnose_fit(model, X_val, y_val, history):
    train_acc = history.history['sparse_categorical_accuracy'][-1]
    val_acc = history.history['val_sparse_categorical_accuracy'][-1]
    gap = train_acc - val_acc

    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    macro_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

    if macro_f1 < 0.5:
        print(f"Model is underfitting, macro-F1 score of {macro_f1:.4f}")
        return "underfit"
    elif gap > 0.15:
        print(f"Model is overfitting, gap of {gap:.4f}")
        return "overfit"
    else:
        print(f"Model is acceptable, macro-F1 score of {macro_f1:.4f} and gap of {gap:.4f}")
        return "acceptable"

def train_until_ready(X, y, seen_classes, n_features, batch, max_hidden=10):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    best = None  # (macro_f1, n_hidden, model, history)
    n_hidden = 0
    while True:
        model = build_model(n_hidden, len(seen_classes), n_features)
        weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        weights = np.sqrt(weights)  # dampens the most extreme weights
        class_weight_dict = dict(zip(np.unique(y_tr), weights))

        history = model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=batch,
                             validation_data=(X_val, y_val), verbose=0)

        val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        macro_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)
        train_acc = history.history['sparse_categorical_accuracy'][-1]
        val_acc = history.history['val_sparse_categorical_accuracy'][-1]
        gap = train_acc - val_acc

        print(f"hidden_units={n_hidden} -> macro_f1={macro_f1:.4f}, gap={gap:.4f}")

        if best is None or macro_f1 > best[0]:
            best = (macro_f1, n_hidden, model, history)

        if macro_f1 >= 0.5 or gap > 0.15 or n_hidden >= max_hidden:
            return best[2], best[3], best[1]   # return the BEST model seen, not the last
        n_hidden += 1

def get_loss(history1):
    final_loss = history1['loss'][-1]
    print(f"Final Training Loss (MSE): {final_loss:.4f}")


# neural network implementation

# model data initialization : --------------------------------------------------------------------------------------------------------------------------------------------

#%% model1 : -------------------------------------------------------------------------------------------------------------------------------------------------------------

class_order = get_class_order(wine_data['quality'])
X_train, X_test, y_train, y_test = train_test(wine_data)
scaled_X_train, scaled_X_test = robust_scaler(X_train, X_test)

seen = class_order 

def baseline(X_train, y_train, X_test, y_test):
    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(30, activation='relu'),
        layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH,
                        validation_split=0.2, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Baseline Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return model, history

all_classes = class_order   # every class, in the same rarity order used throughout
y_s = remap_labels(y_train, all_classes).astype('int32')
y_t = remap_labels(y_test, all_classes).astype('int32')
baseline_model, baseline_history = baseline(scaled_X_train, y_s, scaled_X_test, y_t)


#%% incremental learning : ---------------------------------------------------------------------------------------------------------------------------------------------------
n_stages = len(class_order) -  1   
seen = class_order[:2]            
models_by_stage = []

for i in range(2, len(class_order) + 1):
    stage = i - 2
    X_s, y_s, classes = data_for_stage(scaled_X_train, y_train, stage=stage,
                                        class_order=class_order, n_initial=2, cumulative=True)
    batch_size_for_stage = min(BATCH, max(8, len(X_s) // 4))
    X_s, y_s, classes = data_for_stage(scaled_X_train, y_train, stage=stage,
                                        class_order=class_order, n_initial=2, cumulative=True)
    batch_size_for_stage = min(BATCH, max(8, len(X_s) // 4))
    y_s = remap_labels(y_s, classes).astype('int32')
    X_t, y_t, _ = data_for_stage(scaled_X_test, y_test, stage=stage,
                                  class_order=class_order, n_initial=2, cumulative=True)
    y_t = remap_labels(y_t, classes).astype('int32')

    model, history, n_hidden = train_until_ready(X_s, y_s, classes, n_features=scaled_X_train.shape[1], batch=batch_size_for_stage)

    test_loss, test_acc = model.evaluate(X_t, y_t, verbose=0)

    pred_idx = np.argmax(model.predict(X_t, verbose=0), axis=1)
    macro_f1 = f1_score(y_t, pred_idx, average='macro')
    print(classification_report(y_t, pred_idx, zero_division=0))

    models_by_stage.append({
        "stage": stage,
        "classes": classes.copy(),
        "n_hidden": n_hidden,
        "model": model,
        "history": history,
        "train_loss": history.history['loss'][-1],
        "train_acc": history.history['sparse_categorical_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "val_acc": history.history['val_sparse_categorical_accuracy'][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "X_t": X_t,
        "y_t": y_t,
    })
#%% model summary : ---------------------------------------------------------------------------------------------------------------------------------------------------

summary = pd.DataFrame([{
    "stage": r["stage"],
    "n_classes": len(r["classes"]),
    "classes": r["classes"],
    "n_hidden": r["n_hidden"],
    "train_loss": r["train_loss"],
    "train_acc": r["train_acc"],
    "val_loss": r["val_loss"],
    "val_acc": r["val_acc"],
    "test_loss": r["test_loss"],
    "test_acc": r["test_acc"],
} for r in models_by_stage])

print(summary.to_string(index=False))

# %% findings : ---------------------------------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(summary['stage'], summary['n_hidden'], marker='o')
axes[0].set_xlabel('Stage'); axes[0].set_ylabel('Hidden units'); axes[0].set_title('Architecture growth')

axes[1].plot(summary['stage'], summary['train_loss'], marker='o', label='train')
axes[1].plot(summary['stage'], summary['val_loss'], marker='o', label='val')
axes[1].plot(summary['stage'], summary['test_loss'], marker='o', label='test')
axes[1].set_xlabel('Stage'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].set_title('Loss per stage')

axes[2].plot(summary['stage'], summary['train_acc'], marker='o', label='train')
axes[2].plot(summary['stage'], summary['val_acc'], marker='o', label='val')
axes[2].plot(summary['stage'], summary['test_acc'], marker='o', label='test')
axes[2].set_xlabel('Stage'); axes[2].set_ylabel('Accuracy'); axes[2].legend(); axes[2].set_title('Accuracy per stage')

fig.tight_layout()
fig.savefig('incremental_stage_metrics.png', dpi=150)
plt.show()
# %%

# started with acc 0.6
# went to 0.7 ; test accuracy decreased a lot for earlier stages and increased a little for later stages
# then increased to 0.65, which is a better threshold for underfitting detection. The model now correctly identifies underfitting when training accuracy is below 0.65, allowing for better model performance and generalization.
