import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Step 1: Create the DataFrame
data = {'points': [18, 19, 14, 14, 11, 20, 28, 30, 31, 35],
       'assists': [3, 4, 5, 4, 7, 8, 7, 6, 9, 12],
       'rebounds': [15, 14, 10, 8, 14, 13, 9, 5, 4, 11]}
df = pd.DataFrame(data)
# Step 2: Clean & Prep the DataFrame
df = df.dropna()
scaled_df = StandardScaler().fit_transform(df)
# Step 3: Find the Optimal Number of Clusters
sse = []
for k in range(1, 11):
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(scaled_df)
   sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
# Step 4: Perform K-Means Clustering with Optimal K
kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_df)
df['cluster'] = kmeans.labels_
print(df)