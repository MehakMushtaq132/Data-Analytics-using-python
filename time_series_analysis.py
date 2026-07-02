# ==============================================
# Time Series Analysis using ARIMA
# Dataset: airline_passengers.csv
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ==============================================
# Load Dataset
# ==============================================

df = pd.read_csv("airline_passengers.csv")

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"])

# Set Month as index
df.set_index("Month", inplace=True)

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==============================================
# Plot Original Time Series
# ==============================================

plt.figure(figsize=(12,6))

plt.plot(df["Passengers"], linewidth=2)

plt.title("Monthly Airline Passengers")

plt.xlabel("Year")

plt.ylabel("Number of Passengers")

plt.grid(True)

plt.show()

# ==============================================
# Moving Averages
# ==============================================

df["7_Month_MA"] = df["Passengers"].rolling(window=7).mean()

df["12_Month_MA"] = df["Passengers"].rolling(window=12).mean()

plt.figure(figsize=(12,6))

plt.plot(df["Passengers"], label="Original Data")

plt.plot(df["7_Month_MA"], label="7-Month Moving Average")

plt.plot(df["12_Month_MA"], label="12-Month Moving Average")

plt.title("Moving Average Analysis")

plt.xlabel("Year")

plt.ylabel("Passengers")

plt.legend()

plt.grid(True)

plt.show()

# ==============================================
# Time Series Decomposition
# ==============================================

decomposition = seasonal_decompose(
    df["Passengers"],
    model="additive",
    period=12
)

decomposition.plot()

plt.show()

# ==============================================
# Train Test Split
# ==============================================

train_size = int(len(df) * 0.80)

train = df["Passengers"][:train_size]

test = df["Passengers"][train_size:]

print("\nTraining Data:", len(train))

print("Testing Data :", len(test))

# ==============================================
# Build ARIMA Model
# ==============================================

model = ARIMA(train, order=(5,1,0))

model_fit = model.fit()

print("\nARIMA Model Summary\n")

print(model_fit.summary())

# ==============================================
# Forecast Test Data
# ==============================================

forecast = model_fit.forecast(steps=len(test))

forecast = pd.Series(forecast.values, index=test.index)

# ==============================================
# Evaluation
# ==============================================

mae = mean_absolute_error(test, forecast)

rmse = np.sqrt(mean_squared_error(test, forecast))

print("\nModel Performance")

print("Mean Absolute Error (MAE):", round(mae,2))

print("Root Mean Square Error (RMSE):", round(rmse,2))

# ==============================================
# Actual vs Forecast
# ==============================================

plt.figure(figsize=(12,6))

plt.plot(train, label="Training Data")

plt.plot(test, label="Actual Data")

plt.plot(forecast, label="Forecast", linewidth=3)

plt.title("ARIMA Forecast")

plt.xlabel("Year")

plt.ylabel("Passengers")

plt.legend()

plt.grid(True)

plt.show()

# ==============================================
# Forecast Next 12 Months
# ==============================================

future_forecast = model_fit.forecast(steps=12)

future_dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

future = pd.Series(future_forecast.values, index=future_dates)

print("\nForecast for Next 12 Months\n")

print(future)

plt.figure(figsize=(12,6))

plt.plot(df["Passengers"], label="Historical Data")

plt.plot(future, label="Next 12 Months Forecast", linewidth=3)

plt.title("Future Passenger Forecast")

plt.xlabel("Year")

plt.ylabel("Passengers")

plt.legend()

plt.grid(True)

plt.show()

print("\nTime Series Analysis Completed Successfully!")