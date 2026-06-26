# ==========================================
# Exploratory Data Analysis (EDA)
# Movie Review Dataset
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("IMDB Dataset(1).csv")

# Display First and Last Records
print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

# Dataset Information
print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ----------------------------
# Basic Visualizations
# ----------------------------

# Histogram for Numeric Columns
numeric_columns = df.select_dtypes(include='number')

if not numeric_columns.empty:
    numeric_columns.hist(figsize=(10, 6))
    plt.suptitle("Histogram of Numeric Columns")
    plt.show()

# Bar Chart for First Categorical Column
categorical_columns = df.select_dtypes(include='object')

if not categorical_columns.empty:
    first_cat = categorical_columns.columns[0]

    plt.figure(figsize=(8,5))
    df[first_cat].value_counts().head(10).plot(kind='bar')
    plt.title(f"Top 10 Values in {first_cat}")
    plt.xlabel(first_cat)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Box Plot for Numeric Columns
if not numeric_columns.empty:
    plt.figure(figsize=(8,5))
    numeric_columns.boxplot()
    plt.title("Box Plot of Numeric Columns")
    plt.show()

print("\nEDA Completed Successfully!")