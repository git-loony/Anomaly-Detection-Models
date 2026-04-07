import pandas as pd
import joblib
import torch
import numpy as np
from prophet.serialize import model_from_json
from trains import feature_engineering, rule_based_checks, Autoencoder, features
import sqlite3

conn = sqlite3.connect("models/models.db")
cursor = conn.cursor()

# LOAD MODELS
iso = joblib.load("models/iso.pkl")
kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

with open("models/prophet.json") as f:
    prophet_model = model_from_json(f.read())

model_ae = Autoencoder(len(features))
model_ae.load_state_dict(torch.load("models/ae.pth", weights_only=True))
model_ae.eval()


def predict(file):
    df_raw = pd.read_csv(file)
    
    
    # Use the saved encoders from training
    df, _ = feature_engineering(df_raw, encoders=encoders)
    df = rule_based_checks(df)

    X = df[features]
    X_scaled = scaler.transform(X)

    # 1. Isolation Forest Score
    df["iso_score"] = -iso.score_samples(X_scaled)

    # 2. KMeans Distance
    clusters = kmeans.predict(X_scaled)
    df["cluster"] = clusters
    centroids = kmeans.cluster_centers_
    df["kmeans_distance"] = [
        np.linalg.norm(X_scaled[i] - centroids[clusters[i]])
        for i in range(len(X_scaled))
    ]

    # 3. Prophet (Daily Trend Anomaly)
    daily = df.groupby("order_date")["total_amount"].sum().reset_index()
    daily.columns = ["ds", "y"]
    forecast = prophet_model.predict(daily)
    daily["prophet_score"] = np.where(
        daily["y"] > forecast["yhat_upper"], 
        daily["y"] - forecast["yhat_upper"],
        np.where(daily["y"] < forecast["yhat_lower"], 
                 forecast["yhat_lower"] - daily["y"], 0)
    )
    
    df = df.merge(daily[["ds", "prophet_score"]],
                  left_on="order_date",
                  right_on="ds",
                  how="left").drop(columns=["ds"])
    df["prophet_score"] = df["prophet_score"].fillna(0)

    # 4. Autoencoder Reconstruction Error
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model_ae(X_tensor)
        df["autoencoder_score"] = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    def norm(s):
        if s.max() == s.min(): return np.zeros_like(s)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)    
    
    def prophet_norm(s):
        if s.max() == s.min(): return np.zeros_like(s)
        return (s.max() - s) / (s.max() - s.min() + 1e-9)

    # Ensemble Scoring
    df["final_score"] = (
        0.3 * norm(df["iso_score"]) +
        0.2 * norm(df["kmeans_distance"]) +
        0.15 * prophet_norm(df["prophet_score"]) + 
        0.15 * norm(df["autoencoder_score"]) +
        0.2 * norm(df["rule_score"])
    )

    threshold = df["final_score"].quantile(0.95)
    df["is_anomaly"] = df["final_score"] > threshold
    
    # Prepare Output
    dff = df[["order_id", "customer_id", "final_score", "is_anomaly"]].copy()
    
    outlier_score = norm(df["iso_score"]) 
    prophet_score = prophet_norm(df["prophet_score"])
    peer_group_score = norm(df["kmeans_distance"])
    novelty_score = norm(df["autoencoder_score"])
    rule_score = norm(df["rule_score"])
    
    dff["scores"] = [
        {
            "outlier_score": float(o),
            "prophet_score": float(p),
            "peer_group_score": float(pg),
            "novelty_score": float(n),
            "rule_score": float(r),
        }
        for o, p, pg, n, r in zip(
            outlier_score, prophet_score, peer_group_score, novelty_score, rule_score
        )
    ]
    
    return dff