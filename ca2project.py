import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
df = pd.read_excel("dataset.xlsx")

# Display the first few rows of the dataset
print(df.head())

# Get a summary of the dataset
summary = df.describe(include='all')
print(summary)

# Check for null values
null_values = df.isnull().sum()
print(null_values)

# Fill numerical null values with the mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# Calculate correlation matrix only for numeric columns
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

# Calculate covariance matrix only for numeric columns
covariance_matrix = df[numeric_cols].cov()
print(covariance_matrix)

# Calculate IQR for numerical columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).sum()
print("Outliers based on IQR method:\n", outliers)

df.rename(columns={
    'Central Goods and Services Tax ( CGST ) Revenue': 'CGST',
    'State Goods and Services Tax ( SGST )Revenue': 'SGST',
    'Integrated Goods and Services Tax ( IGST )Revenue': 'IGST',
    'CESS Tax Revenue': 'CESS',
    'srcStateName': 'State',
    'Month': 'MonthFull'
}, inplace=True)

# --- 2. Summary Stats ---
print("\n🔹 Descriptive Statistics:\n")
print(df[['CGST', 'SGST', 'IGST', 'CESS']].describe())

# --- 3. Correlation Heatmap ---
plt.figure(figsize=(8, 6))
corr = df[['CGST', 'SGST', 'IGST', 'CESS']].corr()
sns.heatmap(corr, annot=True, cmap='Greens')
plt.title("Correlation Heatmap of GST Revenues")
plt.tight_layout()
plt.show()

# --- 4. Revenue Distributions ---
plt.figure(figsize=(14, 8))
for i, col in enumerate(['CGST', 'SGST', 'IGST', 'CESS']):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], bins=30, kde=True, color='cyan')
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.show()

state_totals = df.groupby('State')[['CGST', 'SGST', 'IGST', 'CESS']].sum()
state_totals['Total'] = state_totals.sum(axis=1)
top_states = state_totals.sort_values('Total', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_states['Total'], y=top_states.index, palette='pastel')
plt.title("Top 10 States by Total GST Revenue")
plt.xlabel("Total GST Revenue")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# --- 6. Monthly Trends ---
monthly_trend = df.groupby('MonthFull')[['CGST', 'SGST', 'IGST', 'CESS']].sum()
monthly_trend = monthly_trend.reset_index()

plt.figure(figsize=(12, 6))
for tax in ['CGST', 'SGST', 'IGST', 'CESS']:
    sns.lineplot(x='MonthFull', y=tax, data=monthly_trend, label=tax)
plt.title("Monthly GST Revenue Trends")
plt.xticks(rotation=45)
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# --- 7. Overall GST Type Share ---
total_share = df[['CGST', 'SGST', 'IGST', 'CESS']].sum()
plt.figure(figsize=(6, 6))
plt.pie(total_share, labels=total_share.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("muted"))
plt.title("Overall GST Revenue Share by Type")
plt.tight_layout()
plt.show()
