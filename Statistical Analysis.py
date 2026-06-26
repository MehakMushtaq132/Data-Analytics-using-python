# Import Libraries
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# Load Dataset
df = pd.read_csv("movies.csv")

# Select numeric columns
numeric_df = df.select_dtypes(include=['number'])

# ============================================
# 1. Descriptive Statistics
# ============================================

print("===== DESCRIPTIVE STATISTICS =====\n")

print("Mean:")
print(numeric_df.mean())

print("\nMedian:")
print(numeric_df.median())

print("\nVariance:")
print(numeric_df.var())

print("\nStandard Deviation:")
print(numeric_df.std())

print("\nSummary Statistics:")
print(numeric_df.describe())

# ============================================
# 2. Correlation Analysis
# ============================================

print("\n===== CORRELATION MATRIX =====\n")
print(numeric_df.corr())

# ============================================
# 3. Independent T-Test
# ============================================

high_rating = df[df["vote_average"] >= 7]["vote_count"]
low_rating = df[df["vote_average"] < 7]["vote_count"]

t_stat, p_value = ttest_ind(high_rating, low_rating, equal_var=False)

print("\n===== T-TEST =====")
print("T-Statistic =", t_stat)
print("P-Value =", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
else:
    print("Accept Null Hypothesis")

# ============================================
# 4. Chi-Square Test
# ============================================

# Create categories
df["Rating_Category"] = df["vote_average"].apply(
    lambda x: "High" if x >= 7 else "Low"
)

median_votes = df["vote_count"].median()

df["Vote_Category"] = df["vote_count"].apply(
    lambda x: "High" if x >= median_votes else "Low"
)

table = pd.crosstab(df["Rating_Category"], df["Vote_Category"])

chi2, p, dof, expected = chi2_contingency(table)

print("\n===== CHI-SQUARE TEST =====")
print("Chi-square Statistic =", chi2)
print("P-Value =", p)

if p < 0.05:
    print("Variables are Dependent")
else:
    print("Variables are Independent")

# ============================================
# 5. ANOVA
# ============================================

group1 = df[df["vote_average"] < 6]["vote_count"]
group2 = df[(df["vote_average"] >= 6) & (df["vote_average"] < 8)]["vote_count"]
group3 = df[df["vote_average"] >= 8]["vote_count"]

F, p = f_oneway(group1, group2, group3)

print("\n===== ANOVA =====")
print("F-Statistic =", F)
print("P-Value =", p)

if p < 0.05:
    print("Significant Difference Exists")
else:
    print("No Significant Difference")