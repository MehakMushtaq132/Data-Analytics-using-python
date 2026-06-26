# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Dataset
df = pd.read_csv("movies.csv")

# Keep only numeric columns
df = df.select_dtypes(include=['number']).dropna()

# ============================================
# LINEAR REGRESSION
# ============================================

X = df[['vote_count']]
y = df['vote_average']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================================
# Evaluation Metrics
# ============================================

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# ============================================
# 1. Regression Line
# ============================================

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression")
plt.xlabel("Vote Count")
plt.ylabel("Vote Average")
plt.legend()
plt.show()

# ============================================
# 2. Actual vs Predicted
# ============================================

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

# ============================================
# 3. Residual Plot
# ============================================

residuals = y_test - y_pred

plt.figure(figsize=(7,6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ============================================
# 4. Performance Metrics Bar Chart
# ============================================

metrics = ['R² Score', 'MSE', 'RMSE', 'MAE']
values = [r2, mse, rmse, mae]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)
plt.title("Regression Performance")
plt.ylabel("Value")

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.show()

# ============================================
# MULTIPLE LINEAR REGRESSION
# ============================================

X = df.drop('vote_average', axis=1)
y = df['vote_average']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# ============================================
# Multiple Regression Actual vs Predicted
# ============================================

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multiple Linear Regression")
plt.show()

# ============================================
# Performance Metrics
# ============================================

metrics = ['R² Score', 'MSE', 'RMSE', 'MAE']
values = [r2, mse, rmse, mae]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.title("Multiple Regression Performance")
plt.ylabel("Value")
plt.show()