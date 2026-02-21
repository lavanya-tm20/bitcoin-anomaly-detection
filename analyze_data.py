import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")
HIGH_RISK_PATH = os.path.join(PROJECT_ROOT, "data", "high_risk_samples.csv")

# ----------------------------
# Load processed data
# ----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found! Make sure processed_data.csv exists in the data folder.")

df = pd.read_csv(DATA_PATH)

# ----------------------------
# Inspect the data
# ----------------------------
print("First 5 rows of processed data:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

# ----------------------------
# Identify high-risk anomalies
# ----------------------------
RISK_THRESHOLD = 0.3  # Change if needed
high_risk = df[df['risk_score'] > RISK_THRESHOLD]

print(f"\nNumber of high-risk anomalies (risk_score > {RISK_THRESHOLD}): {len(high_risk)}")

if len(high_risk) > 0:
    print("\nHigh-risk anomalies:")
    print(high_risk[['date', 'attack_type', 'sup_pred', 'risk_score', 'ocsvm_score', 'kmeans_distance']])

# Save high-risk samples
high_risk.to_csv(HIGH_RISK_PATH, index=False)
print(f"\nHigh-risk anomalies saved to: {HIGH_RISK_PATH}")

# ----------------------------
# Visualizations
# ----------------------------

# Histogram of risk scores
plt.figure(figsize=(8,5))
plt.hist(df['risk_score'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Risk Score')
plt.ylabel('Count')
plt.title('Distribution of Hybrid Risk Scores')
plt.tight_layout()
plt.show()

# Scatter plot: KMeans distance vs OCSVM score colored by risk
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df['kmeans_distance'], df['ocsvm_score'], 
    c=df['risk_score'], cmap='coolwarm', s=50, alpha=0.7
)
plt.colorbar(scatter, label='Risk Score')
plt.xlabel('KMeans Distance')
plt.ylabel('OCSVM Score')
plt.title('Anomaly Analysis: KMeans Distance vs OCSVM Score')
plt.tight_layout()
plt.show()
