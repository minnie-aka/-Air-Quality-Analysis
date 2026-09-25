# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, shapiro
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score


# LOAD DATASET

df = pd.read_csv("AIR_QUALITY_INDEX_cleaned.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)


# DATA CLEANING

print("\nMissing Values:\n", df.isnull().sum())

# Fill numeric columns with 0
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(0)

# Fill categorical columns with 'Unknown'
cat_cols = df.select_dtypes(exclude=[np.number]).columns
df[cat_cols] = df[cat_cols].fillna('Unknown')

df.drop_duplicates(inplace=True)


# FEATURE ENGINEERING (example: total pollution index if multiple pollutants exist)
if len(num_cols) >= 2:
    df['TOTAL_POLLUTION'] = df[num_cols].sum(axis=1)


# DESCRIPTIVE STATISTICS

print("\nSummary Statistics:\n", df.describe())


# VISUALIZATION (EDA)

# Distribution
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# KDE
for col in num_cols:
    sns.kdeplot(df[col], fill=True)
    plt.title(f"KDE Plot - {col}")
    plt.show()

# Pairplot (if multiple numeric columns)
if len(num_cols) > 2:
    sns.pairplot(df[num_cols])
    plt.show()

# Boxplot
sns.boxplot(data=df[num_cols])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Scatter (first two numeric columns)
if len(num_cols) >= 2:
    sns.scatterplot(x=num_cols[0], y=num_cols[1], data=df)
    plt.title(f"{num_cols[0]} vs {num_cols[1]}")
    plt.show()

# Regression Plot
if len(num_cols) >= 2:
    sns.regplot(x=num_cols[0], y=num_cols[1], data=df)
    plt.title("Regression Plot")
    plt.show()

# Heatmap
sns.heatmap(df[num_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# OUTLIER REMOVAL (IQR on first numeric column)

if len(num_cols) > 0:
    col = num_cols[0]
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]


# CORRELATION
print("\nCorrelation:\n", df.corr(numeric_only=True))


# STATISTICAL TESTING

if len(num_cols) >= 1:
    sample_col = num_cols[0]
    sample_size = min(5000, len(df))
    stat, p = shapiro(df[sample_col].sample(sample_size))
    print("\nShapiro Test p-value:", p)

if len(num_cols) >= 2:
    t_stat, p_val = ttest_ind(df[num_cols[0]], df[num_cols[1]])
    print("T-Test p-value:", p_val)


# LINEAR REGRESSION MODEL

if len(num_cols) >= 3:
    X = df[[num_cols[0], num_cols[1]]]
    y = df[num_cols[2]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nR2 Score:", r2_score(y_test, y_pred))


# CLASSIFICATION (BONUS)

if len(num_cols) >= 1:
    threshold = df[num_cols[0]].mean()
    df['AQI_Category'] = np.where(df[num_cols[0]] > threshold, 'High', 'Low')

    X_cls = df[num_cols[:2]] if len(num_cols) >= 2 else df[[num_cols[0]]]
    y_cls = df['AQI_Category']

    clf = DecisionTreeClassifier()
    clf.fit(X_cls, y_cls)


# SAVE FINAL DATA

df.to_csv("final_aqi_data.csv", index=False)

print("AQI PROJECT COMPLETED SUCCESSFULLY")
