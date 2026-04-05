import pandas as pd
import joblib
import torch
import os

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

from prophet.serialize import model_to_json



from trains import feature_engineering, rule_based_checks, Autoencoder, features

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/raw_dataset.csv")

df, encoders = feature_engineering(df)
df = rule_based_checks(df)

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
iso = IsolationForest(contamination=0.05).fit(X_scaled)
kmeans = KMeans(n_clusters=5, n_init=10).fit(X_scaled)

# Prophet
daily = df.groupby("order_date")["total_amount"].sum().reset_index()
daily.columns = ["ds", "y"]
daily = daily.sort_values("ds")

prophet_model = Prophet()
prophet_model.fit(daily)

# Autoencoder
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
model_ae = Autoencoder(X_scaled.shape[1])

optimizer = torch.optim.Adam(model_ae.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for _ in range(10):
    optimizer.zero_grad()
    loss = criterion(model_ae(X_tensor), X_tensor)
    loss.backward()
    optimizer.step()

# SAVE
joblib.dump(iso, "models/iso.pkl")
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

with open("models/prophet.json", "w") as f:
    f.write(model_to_json(prophet_model))
torch.save(model_ae.state_dict(), "models/ae.pth")

print("✅ Models saved!")