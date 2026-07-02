# Principal Component Analysis (PCA)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data.csv")   # Change filename if needed

print("First 5 Rows:")
print(df.head())

# -------------------------------
# Select Numerical Columns
# -------------------------------
numeric_df = df.select_dtypes(include=['number'])

# Fill missing values with column mean
numeric_df = numeric_df.fillna(numeric_df.mean())

# -------------------------------
# Standardize Data
# -------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# -------------------------------
# Apply PCA
# -------------------------------
pca = PCA()

principal_components = pca.fit_transform(scaled_data)

# -------------------------------
# Explained Variance
# -------------------------------
explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i+1}: {var:.4f}")

# -------------------------------
# Cumulative Explained Variance
# -------------------------------
cumulative_variance = explained_variance.cumsum()

print("\nCumulative Explained Variance:")
for i, var in enumerate(cumulative_variance):
    print(f"PC1 to PC{i+1}: {var:.4f}")

# -------------------------------
# Scree Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1),
         explained_variance,
         marker='o',
         linestyle='--')

plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()

# -------------------------------
# PCA with 2 Components
# -------------------------------
pca2 = PCA(n_components=2)
pca_result = pca2.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    pca_result,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nFirst 5 PCA Results:")
print(pca_df.head())

# -------------------------------
# Scatter Plot
# -------------------------------
plt.figure(figsize=(8,6))

plt.scatter(
    pca_df["Principal Component 1"],
    pca_df["Principal Component 2"],
    alpha=0.7
)

plt.title("PCA - First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------
# PCA with 3 Components
# ---------------------------------
pca3 = PCA(n_components=3)
pca_result3 = pca3.fit_transform(scaled_data)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pca_result3[:,0],
    pca_result3[:,1],
    pca_result3[:,2],
    alpha=0.7
)

ax.set_title("3D PCA Visualization")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

plt.show()