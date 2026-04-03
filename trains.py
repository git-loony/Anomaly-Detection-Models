# =========================================
# 1. IMPORTS
# =========================================
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

from prophet import Prophet

import torch
import torch.nn as nn


# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_csv("data/raw_dataset.csv")


# =========================================
# 3. FEATURE ENGINEERING
# =========================================
def feature_engineering(df):
    df = df.copy()

    # Monetary
    df["total_amount"] = df["price"] * df["quantity"]
    df["log_amount"] = np.log1p(df["total_amount"])

    # Customer behavior
    df["order_count"] = df.groupby("customer_id")["order_id"].transform("count")
    df["avg_spend_customer"] = df.groupby("customer_id")["total_amount"].transform("mean")
    df["std_spend_customer"] = df.groupby("customer_id")["total_amount"].transform("std").fillna(0)

    # Time
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Encode categorical
    cat_cols = ["payment_method", "country", "city", "category"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


df, encoders = feature_engineering(df)


# =========================================
# 4. RULE-BASED CHECKS
# =========================================
def rule_based_checks(df):
    df = df.copy()

    # Duplicate
    df["rule_duplicate"] = df.duplicated(
        subset=["customer_id", "product_id", "order_date", "total_amount"],
        keep=False
    ).astype(int)

    # Price spike vs category
    category_avg = df.groupby("category")["total_amount"].transform("mean")
    df["rule_price_spike"] = (df["total_amount"] > 3 * category_avg).astype(int)

    # Quantity anomaly
    df["rule_quantity"] = (df["quantity"] > df["quantity"].quantile(0.99)).astype(int)

    # Rare payment
    payment_freq = df["payment_method"].value_counts(normalize=True)
    df["rule_payment"] = df["payment_method"].map(payment_freq) < 0.02
    df["rule_payment"] = df["rule_payment"].astype(int)

    # Geo mismatch
    customer_country = df.groupby("customer_id")["country"].transform(lambda x: x.mode()[0])
    df["rule_geo"] = (df["country"] != customer_country).astype(int)

    # Known fraud
    if "is_fraud" in df.columns:
        df["rule_known_fraud"] = df["is_fraud"].astype(int)
    else:
        df["rule_known_fraud"] = 0

    # Weighted rule score
    weights = {
        "rule_duplicate": 2,
        "rule_price_spike": 2,
        "rule_quantity": 1,
        "rule_payment": 1,
        "rule_geo": 1,
        "rule_known_fraud": 3
    }

    df["rule_score"] = sum(df[col] * w for col, w in weights.items())

    return df


df = rule_based_checks(df)


# =========================================
# 5. FEATURE SELECTION
# =========================================
features = [
    "price", "quantity", "total_amount", "log_amount",
    "order_count", "avg_spend_customer", "std_spend_customer",
    "day_of_week", "is_weekend",
    "payment_method", "country", "city", "category"
]

X = df[features].copy()


# =========================================
# 6. SCALING
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =========================================
# 7. ISOLATION FOREST
# =========================================
iso = IsolationForest(contamination=0.05, random_state=42)
df["iso_score"] = -iso.fit_predict(X_scaled)


# =========================================
# 8. K-MEANS
# =========================================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters

centroids = kmeans.cluster_centers_

df["kmeans_distance"] = [
    np.linalg.norm(X_scaled[i] - centroids[clusters[i]])
    for i in range(len(X_scaled))
]


# =========================================
# 9. PROPHET
# =========================================
daily = df.groupby("order_date")["total_amount"].sum().reset_index()
daily.columns = ["ds", "y"]

model_prophet = Prophet()
model_prophet.fit(daily)

forecast = model_prophet.predict(daily)

daily["prophet_score"] = np.abs(daily["y"] - forecast["yhat"])

df = df.merge(
    daily[["ds", "prophet_score"]],
    left_on="order_date",
    right_on="ds",
    how="left"
).drop(columns=["ds"])


# =========================================
# 10. AUTOENCODER
# =========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

model_ae = Autoencoder(X_scaled.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_ae.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model_ae(X_tensor)
    loss = criterion(outputs, X_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    reconstructed = model_ae(X_tensor)
    errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

df["autoencoder_score"] = errors.numpy()


# =========================================
# 11. NORMALIZATION
# =========================================
def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

df["iso_n"] = normalize(df["iso_score"])
df["kmeans_n"] = normalize(df["kmeans_distance"])
df["prophet_n"] = normalize(df["prophet_score"])
df["ae_n"] = normalize(df["autoencoder_score"])
df["rule_n"] = normalize(df["rule_score"])


# =========================================
# 12. FINAL SCORE
# =========================================
df["final_score"] = (
    0.30 * df["iso_n"] +
    0.20 * df["kmeans_n"] +
    0.15 * df["prophet_n"] +
    0.15 * df["ae_n"] +
    0.20 * df["rule_n"]
)


# =========================================
# 13. ANOMALY FLAG
# =========================================
threshold = df["final_score"].quantile(0.95)
df["is_anomaly"] = df["final_score"] > threshold


# =========================================
# 14. OUTPUT
# =========================================
print(df[[
    "order_id",
    "final_score",
    "is_anomaly",
    "rule_score",
    "iso_n",
    "kmeans_n",
    "prophet_n",
    "ae_n"
]].head())

df.to_csv("anomaly_results.csv", index=False)