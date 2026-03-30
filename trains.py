# full_billing_anomaly_pipeline_with_eval.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from prophet import Prophet
import sqlite3
import joblib
import matplotlib.pyplot as plt

# ===========================
# 0. Create OUTMODEL folder
# ===========================
MODEL_DIR = "MODEL_DB"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================
# 1. Load Data
# ===========================
df = pd.read_csv("data/raw_dataset.csv")
df['order_date'] = pd.to_datetime(df['order_date'])

# ===========================
# 2. Feature Engineering
# ===========================
features = [
    'price', 'quantity', 'total_amount',
    'user_txn_count', 'user_avg_amount', 'user_max_amount',
    'user_fraud_rate', 'product_fraud_rate', 'product_avg_price',
    'country_fraud_rate', 'payment_fraud_rate'
]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 3. Autoencoder Model
# ===========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

input_dim = X_scaled.shape[1]
autoencoder = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
X_tensor = torch.FloatTensor(X_scaled)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Autoencoder
epochs = 50
for epoch in range(epochs):
    for batch in dataloader:
        data = batch[0]
        output = autoencoder(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    reconstructed = autoencoder(X_tensor).numpy()
    ae_scores = np.mean((X_scaled - reconstructed)**2, axis=1)

# ===========================
# 4. K-Means + Isolation Forest
# ===========================
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

if_scores = np.zeros(X_scaled.shape[0])
isolation_models = {}
for cluster_id in range(n_clusters):
    idx = np.where(clusters == cluster_id)[0]
    X_cluster = X_scaled[idx]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_cluster)
    if_scores[idx] = -iso.score_samples(X_cluster)
    isolation_models[cluster_id] = iso

# ===========================
# 5. Prophet Time-Series per Customer
# ===========================
prophet_scores = np.zeros(X_scaled.shape[0])
for customer_id, group in df.groupby('customer_id'):
    # Prepare Prophet dataframe
    ts = group[['order_date', 'total_amount']].rename(columns={'order_date':'ds', 'total_amount':'y'})
    if len(ts) < 5:
    	# too few points for Prophet
        continue
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(ts)
    forecast = model.predict(ts)
    residual = np.abs(ts['y'].values - forecast['yhat'].values) / (forecast['yhat'].values + 1e-5)
    idx = group.index
    prophet_scores[idx] = residual

# Normalize Prophet scores
prophet_scores = prophet_scores / (prophet_scores.max() + 1e-8)

# ===========================
# 6. Rule-Based Checks
# ===========================
rule_scores = np.zeros(X_scaled.shape[0])
rule_scores += (df['total_amount'] > 3 * df['user_avg_amount']).astype(int)
rule_scores += (df['quantity'] > 10).astype(int)
rule_scores = np.clip(rule_scores, 0, 1)

# ===========================
# 7. Weighted Ensemble
# ===========================
weights = {
    'autoencoder': 0.3,
    'if_kmeans': 0.3,
    'prophet': 0.2,
    'rule': 0.2
}

final_score = (
    weights['autoencoder'] * (ae_scores / (ae_scores.max() + 1e-8)) +
    weights['if_kmeans'] * (if_scores / (if_scores.max() + 1e-8)) +
    weights['prophet'] * prophet_scores +
    weights['rule'] * rule_scores
)

confidence = pd.cut(final_score, bins=[0,0.5,0.8,1.0], labels=['Low','Medium','High'])

# ===========================
# 8. Save Results & Models
# ===========================
df['ae_score'] = ae_scores
df['if_score'] = if_scores
df['prophet_score'] = prophet_scores
df['rule_score'] = rule_scores
df['final_score'] = final_score
df['confidence'] = confidence

# Save to SQLite
db_path = os.path.join(MODEL_DIR, "billing_anomaly.db")
conn = sqlite3.connect(db_path)
df.to_sql("billing_anomalies", conn, if_exists='replace', index=False)
conn.close()
print(f"Data saved to {db_path}")

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Scaler saved")

# Save Autoencoder
torch.save(autoencoder.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pth"))
print("Autoencoder saved")

# Save KMeans
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))
print("KMeans saved")

# Save Isolation Forests per cluster
for cluster_id, iso_model in isolation_models.items():
    joblib.dump(iso_model, os.path.join(MODEL_DIR, f"isolation_forest_cluster_{cluster_id}.pkl"))
print("Isolation Forest models saved")
print("All models, scaler, and DB saved in MODEL folder.")

# ===========================
# 9. Evaluation & Visualization
# ===========================
def plot_score_distribution(scores, title="Score Distribution"):
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()

# Plot each method and ensemble
plot_score_distribution(ae_scores, "Autoencoder Scores")
plot_score_distribution(if_scores, "Isolation Forest Scores")
plot_score_distribution(prophet_scores, "Prophet Residuals")
plot_score_distribution(final_score, "Final Ensemble Scores")

# Optional: print top anomalies
top_n = 10
top_anomalies = df.sort_values("final_score", ascending=False).head(top_n)
print("Top anomalies:")
print(top_anomalies[['customer_id','order_date','total_amount','final_score','confidence']])