#%%
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# TASK 1
#%%
bad_data = pd.read_csv("badDataSet.csv")
bad_data = pd.DataFrame(bad_data)

bad_data.columns = [f"A{i}" for i, _ in enumerate(bad_data.columns, start=1)]
bad_data.rename(columns={bad_data.columns[-1]: "T"}, inplace=True)
target = bad_data["T"]
bad_data.drop("T", axis=1, inplace=True)

# Question 1

#%% replacing random strings in numerical columns with NaN

categorical_cols = []
continuous_cols = []
pd.set_option('display.float_format', '{:.2f}'.format)

for col in bad_data.columns:
    if bad_data[col].dtype == 'str':
        converted = pd.to_numeric(bad_data[col], errors='coerce')
        total = len(bad_data[col])
        convertible = converted.notna()

        pct_convertible = total / convertible.sum() if convertible.sum() > 0 else 0

        if pct_convertible == 0:
            # Case 1: all strings, no numeric values at all -> stays categorical
            categorical_cols.append(col)
            continue
        else :
            # mostly numeric or all numeric, a few stray strings -> clean and convert
            bad_data[col] = converted.astype(float)

    n_unique = bad_data[col].nunique()
    values = bad_data[col].dropna()
    is_ordinal = (values < 10)

    if n_unique <= 10 and is_ordinal.all():
        categorical_cols.append(col)
    # elif n_unique == len(bad_data[col]):
    #     categorical_cols.append(col)
    else:
        continuous_cols.append(col)


print("Categorical:", categorical_cols)
print("Continuous:", continuous_cols)
        

# could remove A17 and A18 since they are just 0's and just 1's ; ie non-informative
# A12 have very random numbers .. could be RGB but ... dont know? 


# %%

# Continuous: 
# Feature, Count, % Miss., Card., Min., 1st Qrt., Mean, Median, 3rd Qrt., Max., Std.Dev.

cont_data = bad_data[continuous_cols]

dqr_cts = pd.DataFrame({
    "Feature": cont_data.columns,
    "Count": cont_data.count(),
    "% Miss.": ((cont_data.isnull().sum() / len(cont_data)) * 100).round(2),
    "Card": cont_data.nunique(),
    "Min.": cont_data.min(),
    "1st Qrt.": cont_data.quantile(0.25),
    "Mean": cont_data.mean(),
    "Median": cont_data.median(),
    "3rd Qrt.": cont_data.quantile(0.75),
    "Max.": cont_data.max(),
    "Std.Dev.": cont_data.std()
})

# Save the DataFrame to a CSV file
dqr_cts.to_csv("dqr_continuous.csv", index=False, header=False)


# %%

# Categorical : 
# Feature, Count, % Miss., Card., Mode, Mode Freq., Mode %, 2nd Mode, 2nd Mode, Freq., 2nd Mode %

cat_data = bad_data[categorical_cols]

dqr_cat = pd.DataFrame({
    "Feature": cat_data.columns,
    "Count": cat_data.count(),
    "% Miss.": ((cat_data.isnull().sum() / len(cat_data)) * 100).round(2),
    "Card": cat_data.nunique(),
    "Mode": cat_data.mode().iloc[0],
    "Mode Freq.": cat_data.apply(lambda x: x.value_counts().iloc[0]),
    "Mode %": (cat_data.apply(lambda x: x.value_counts(normalize=True).iloc[0]) * 100).round(2),
    "2nd Mode": cat_data.apply(lambda x: x.value_counts().index[1] if len(x.value_counts()) > 1 else None),
    "2nd Mode Freq.": cat_data.apply(lambda x: x.value_counts().iloc[1] if len(x.value_counts()) > 1 else None),
    "2nd Mode %": (cat_data.apply(lambda x: x.value_counts(normalize=True).iloc[1] * 100 if len(x.value_counts()) > 1 else None)).round(2)
})

dqr_cat.to_csv("dqr_categorical.csv", index=False, header=False)

#%% 
# target : 

vc = target.value_counts()
vc_norm = target.value_counts(normalize=True)

dqr_target = pd.DataFrame({
    "Feature": [target.name],
    "Count": [target.count()],
    "% Miss.": [round((target.isnull().sum() / len(target)) * 100, 2)],
    "Card": [target.nunique()],
    "Mode": [target.mode().iloc[0]],
    "Mode Freq.": [vc.iloc[0]],
    "Mode %": [round(vc_norm.iloc[0] * 100, 2)],
    "2nd Mode": [vc.index[1] if len(vc) > 1 else None],
    "2nd Mode Freq.": [vc.iloc[1] if len(vc) > 1 else None],
    "2nd Mode %": [round(vc_norm.iloc[1] * 100, 2) if len(vc) > 1 else None]
})

dqr_target.to_csv("dqr_target.csv", index=False, header=False)


# Quesion 2

# histograms for all the continuous descriptive features, and bar plots for all the 
# categorical descriptive features

#%%

# 2. Create a grid of subplots
ncols = 3
nrows = (len(cont_data.columns) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows))
axes = axes.flatten()

# 3. Loop and plot histograms for each numeric feature
for ax, col in zip(axes, cont_data.columns):
    sns.histplot(
        data=cont_data,
        x=col,
        bins=20,
        color='skyblue',
        edgecolor='black',
        ax=ax
    )
    ax.set_title(col, fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

# 4. Remove any empty subplots
for ax in axes[len(cont_data.columns):]:
    fig.delaxes(ax)

# 5. Fix layout and display the plots
plt.tight_layout()
plt.show()

plt.savefig("histograms.pdf", dpi=300, bbox_inches='tight')

# %%

# BARPLOTS
# 1. Setup your categorical feature columns
categorical_features = bad_data[categorical_cols]

# 2. Create a grid of subplots
ncols =2
nrows = (len(categorical_features.columns) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows), sharex=False, sharey=False)
axes = axes.flatten()

# 3. Loop and plot bar plots for each categorical feature
for ax, col in zip(axes, categorical_features.columns):
    plot_data = categorical_features[col].value_counts(dropna=False).reset_index()
    plot_data.columns = [col, 'count']
    plot_data = plot_data.sort_values(by='count', ascending=False)

    sns.barplot(
        data=plot_data,
        x=col,
        y='count',
        ax=ax,
        color='skyblue',
        edgecolor='black'
    )
    ax.set_title(col, fontsize=10)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

    # Make x-axis labels more readable for crowded categorical variables like A2
    if categorical_features[col].nunique() > 20:
        ax.tick_params(axis='x', rotation=90, labelsize=6)
    else :
        ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.set_xticklabels([str(x) for x in plot_data[col]], rotation=90, ha='right')

# 4. Remove any empty subplots
for ax in axes[len(categorical_features.columns):]:
    fig.delaxes(ax)

# 5. Fix layout and display the plots
plt.tight_layout()
plt.show()

plt.savefig("barplots.pdf", dpi=300, bbox_inches='tight')

# %%
# target variable distribution

sns.histplot(
    data=target, 
    bins=20, 
    color='skyblue', 
    edgecolor='black',
    )

plt.title("Target Variable", fontsize=10)
plt.savefig("target_distribution.pdf", dpi=300, bbox_inches='tight')


#%%
# data quality issues report

feature = []
issue = []
evidence = []
handle = []
just = []

def check_card(row):
    if row["Card"] / row["Count"] > 0.95 :
        feature.append(row["Feature"])
        issue.append("Cardinality")
        evidence.append("Cardinality > 95%")
        handle.append("Remove the feature")
        just.append("Near uniqueness of observations suggest some time of identifier, not a predictive feature")

    elif row["Card"] / row["Count"] < 0.20 :
        feature.append(row["Feature"])
        issue.append("Cardinality")
        evidence.append("Cardinality < 5%")
        handle.append("Encode the features using integers")
        just.append("Too many categories for one-hot encoding")

    elif row["Card"] == 1:
        feature.append(row["Feature"])
        issue.append("No Information")
        evidence.append("Cardinality is 1")
        handle.append("Remove the feature")
        just.append("Feature is likely an identifier and does not hold any predictive information")     

def safe_div(a, b):
    return a / b if b not in (0, None) and not pd.isna(b) else np.nan

def check_skew(row):
    median = row["Median"]
    mean = row["Mean"]
    std = row["Std.Dev."]

    skew = 3*(mean-median)/std

    if abs(skew) > 1 :
        feature.append(row["Feature"]) 
        issue.append("Skewness")
        evidence.append(f"Skewness = {skew:.2f} (|skew| > 1 = highly skewed)")
        handle.append("Apply a log transformation to the feature")
        just.append("Log transformation will reduce skewness and make the distribution more symmetric")

    elif abs(skew) < 0.5:
        feature.append(row["Feature"])
        issue.append("Skewness")
        evidence.append(f"Skewness = {skew:.2f} (moderate skew)")
        handle.append("Do nothing")
        just.append("Moderate skew is common and usually not severe enough to require transformation")

def check_mag(row):
    median = row["Median"]
    mean = row["Mean"]
    std = row["Std.Dev."]

    range_ratio = safe_div(row["Max."] - row["Min."], abs(median))
    if range_ratio is not np.nan and range_ratio > 10:
        feature.append(row["Feature"]) 
        issue.append("Magnitude")
        evidence.append("Maximum value is more than 10 times the median")
        handle.append("Rescale the feature")
        just.append("Wide-magnitude features can dominate distance/gradient-based models unless brought to a comparable scale")

    mean_median_pct = safe_div(abs(row["Mean"] - row["Median"]), row["Max."] - row["Min."]) * 100
    if mean_median_pct is not np.nan and mean_median_pct > 20: 
        feature.append(row["Feature"])
        issue.append("Skewness")
        evidence.append("Mean is far higher than median (more than 20% of the range)")
        handle.append("Apply a log transformation")
        just.append("Log transformation will reduce skewness and make the distribution more symmetric")


def check_missing(row):
    miss = row["% Miss."]
    if miss > 0.5:
        feature.append(row["Feature"])
        issue.append("Missingness")
        evidence.append(f"{miss*100}, % missingness")
        handle.append("Remove the feature")
        just.append("Over 50% missing - too little data to impute values reliably without introducing bias")

    elif 0 < miss < 0.5:
        feature.append(row["Feature"])
        issue.append("Missingness")
        evidence.append(f"{miss*100}, % missingness")
        handle.append("Impute the values with the mean")
        just.append("Feature has enough data to reliably impute the values")

def check_robust(row):
    q1 = row["1st Qrt."]
    q3 = row["3rd Qrt."]
    iqr = q3-q1
    ratio = safe_div(iqr, row["Std.Dev."])

    if not pd.isna(ratio) and ratio < 1.0:  
        # IQR is notably smaller than expected relative to std
        # → std is inflated, likely due to outliers/heavy tails
        feature.append(row["Feature"])
        issue.append("Robustness")
        evidence.append(f"IQR/std ratio = {ratio:.2f}, well below the ~1.35 expected for normal data — std likely inflated by outliers")
        handle.append("Apply a log transformation to the feature")
        just.append("There is a heavy tail - log transformation will stabilise the spread")

def check_outliers(row):
    q1 = row["1st Qrt."]
    q3 = row["3rd Qrt."]
    iqr = q3-q1
    check1 = q1 - 1.5*iqr 
    check2 = q3 + 1.5*iqr
    
    feat = row["Feature"]
    outliers_below = bad_data[feat] < check1
    outliers_above = bad_data[feat] > check2
    pct_outliers = (sum(outliers_below) + sum(outliers_above)) / row["Count"] * 100
    
    if pct_outliers < 0.5:
        feature.append(row["Feature"])
        issue.append("Outliers")
        evidence.append(f"Number of outliers {pct_outliers}% is greater than 50%")
        handle.append("Remove the instances")
        just.append("A large number of points are extreme - removing them avoids distorting the model")
    elif pct_outliers > 0:
        feature.append(row["Feature"])
        issue.append("Outliers")
        evidence.append(f"Number of outliers {pct_outliers}% is less than 50%")
        handle.append("Clamp the instances")
        just.append("A small number of points are extreme - capping will preserve the data while limiting the influence")

# categorical checks 

def check_mode(row):
    mode = row["Mode %"]
    if row["Mode %"] > 50:
        feature.append(row["Feature"])
        issue.append("No information")
        evidence.append(f"Mode makes up, {mode}% of the entire observations of this feature")
        handle.append("Remove the feature")
        just.append("There is no predictive information and imputing values will likely be biased")


dqr_cts.apply(check_card, axis=1)
dqr_cts.apply(check_skew, axis=1)
dqr_cts.apply(check_mag, axis=1)
dqr_cts.apply(check_missing, axis=1)
dqr_cts.apply(check_robust, axis=1)
dqr_cts.apply(check_outliers, axis=1)

dqr_cat.apply(check_card, axis=1)
dqr_cat.apply(check_mode, axis=1)
dqr_cat.apply(check_missing, axis=1)



report = pd.DataFrame({
    "Feature" : feature,
    "Data Quality Issue" : issue,
    "Evidence" : evidence, 
    "Handling Strategy" : handle, 
    "Justification": just
})

report.to_csv("dqreport.csv", index=False, header=True)


# CARDINALITY: number of unique values in a column
# - too high (e.g. unique ID-like column with 95%+ unique values in a 
#   categorical field) -> may not be useful as a categorical feature
# - too low (e.g. a column with only 1 unique value) -> no information (see below)
# check: df['col'].nunique() / len(df)

# CORRELATION: relationship between two numeric variables
# - very high correlation (e.g. > 0.9) between two features -> redundancy,
#   multicollinearity risk in models
# - very high correlation between a feature and the target -> possible leakage
# check: df.corr()

# DATA INCONSISTENCY: same real-world entity/value represented differently
# - "NY" vs "New York" vs "ny"; "Male" vs "M" vs "male"
# - conflicting values across related columns (age=5, but job="engineer")
# check: df['col'].unique() -- eyeball for near-duplicate categories

# INDEPENDENT: whether a feature actually contributes unique information
# - if a column can be perfectly derived from other columns (e.g. total = 
#   price * quantity), it's not independent -> redundant
# check: look for deterministic relationships between columns

# INSTANCE DUPLICATION: repeated rows (fully or mostly identical)
# check: df.duplicated().sum()
# also check near-duplicates (duplicated on key columns only):
# df.duplicated(subset=['id']).sum()

# NO INFORMATION: column that doesn't help distinguish rows
# - single constant value across all rows, or (near) all unique (like an ID)
# check: df['col'].nunique() == 1  -->  no information
# df['col'].nunique() == len(df)   -->  likely an identifier, not a feature

# NOT ROBUST: statistics that are highly sensitive to outliers/small changes
# - e.g. mean and std are "not robust" (skewed by outliers), 
#   whereas median and IQR are more robust
# - a "not robust" feature/metric changes drastically if you drop a few rows
# check: compare mean vs median, std vs IQR -- big gaps = fragile stats


# OUTLIERS: as you noted, values far from the rest of the distribution
# check (IQR method):
# Q1, Q3 = df['col'].quantile([0.25, 0.75])
# IQR = Q3 - Q1
# outliers = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)]
# check (z-score method): abs((df['col'] - df['col'].mean()) / df['col'].std()) > 3

# POOR REPRESENTATION: the dataset (or a subgroup within it) doesn't 
# adequately reflect the



# %%
