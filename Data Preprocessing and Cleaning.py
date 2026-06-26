# ----------------------------
# Data Cleaning
# ----------------------------
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
#Return a new Data Frame with no empty cells
new_df = df.dropna()
print(new_df.to_string())
#Remove all rows with NULL values
df.dropna(inplace = True)
print(df.to_string())
#Calculate the MEAN, and replace any empty values with it
x = df["Calories"].mean()
df.fillna({"Calories": x}, inplace=True)
#Calculate the MEDIAN, and replace any empty values with it
x = df["Calories"].median()
df.fillna({"Calories": x}, inplace=True)
#Calculate the MODE, and replace any empty values with it
x = df["Calories"].mode()[0]
df.fillna({"Calories": x}, inplace=True)
print(df.duplicated())
df.drop_duplicates(inplace = True)
# ----------------------------
# Data Preprocessing
# ----------------------------
# ===============================
# Data Preprocessing using Pandas
# ===============================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data(6).csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Shape of dataset
print("\nShape of Dataset:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates (if any)
df = df.drop_duplicates()

# ---------------------------------
# Data Cleaning
# ---------------------------------

# Correct spelling mistakes
df["Car"] = df["Car"].replace({
    "Toyoty": "Toyota",
    "Hundai": "Hyundai"
})

print("\nUnique Car Names:")
print(df["Car"].unique())

# ---------------------------------
# Encode Categorical Data
# ---------------------------------

car_encoder = LabelEncoder()
model_encoder = LabelEncoder()

df["Car"] = car_encoder.fit_transform(df["Car"])
df["Model"] = model_encoder.fit_transform(df["Model"])

print("\nEncoded Dataset:")
print(df.head())

# ---------------------------------
# Feature Scaling
# ---------------------------------

scaler = StandardScaler()

numerical_columns = ["Volume", "Weight", "CO2"]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nScaled Dataset:")
print(df.head())

# ---------------------------------
# Save Preprocessed Dataset
# ---------------------------------

df.to_csv("preprocessed_data.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Preprocessed dataset saved as 'preprocessed_data.csv'")


