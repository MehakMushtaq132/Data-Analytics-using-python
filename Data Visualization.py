import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("movies.csv")

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["vote_average"], bins=10, edgecolor="black")
plt.title("Distribution of Movie Ratings")
plt.xlabel("Vote Average")
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.figure(figsize=(6,5))
sns.boxplot(y=df["vote_average"])
plt.title("Box Plot")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df["vote_count"], df["vote_average"])
plt.xlabel("Vote Count")
plt.ylabel("Vote Average")
plt.title("Vote Count vs Vote Average")
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

print(corr)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# Pair Plot
sns.pairplot(numeric_df)
plt.show()