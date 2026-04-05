import pandas as pd
import joblib
import torch
import numpy as np

from prophet.serialize import model_from_json
from trains import feature_engineering, rule_based_checks, Autoencoder, features

# LOAD MODELS
iso = joblib.load("models/iso.pkl")
kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/prophet.json") as f:
    prophet_model = model_from_json(f.read())

model_ae = Autoencoder(len(features))
model_ae.load_state_dict(torch.load("models/ae.pth"))
model_ae.eval()

def predict(file):
    df = pd.read_csv(file)

    df, _ = feature_engineering(df)
    df = rule_based_checks(df)

    X = df[features]
    X_scaled = scaler.transform(X)

    df["iso_score"] = -iso.score_samples(X_scaled)

    clusters = kmeans.predict(X_scaled)
    df["cluster"] = clusters

    centroids = kmeans.cluster_centers_
    df["kmeans_distance"] = [
        np.linalg.norm(X_scaled[i] - centroids[clusters[i]])
        for i in range(len(X_scaled))
    ]

    daily = df.groupby("order_date")["total_amount"].sum().reset_index()
    daily.columns = ["ds", "y"]

    forecast = prophet_model.predict(daily)
    daily["prophet_score"] = abs(daily["y"] - forecast["yhat"])

    df = df.merge(daily[["ds", "prophet_score"]],
                  left_on="order_date",
                  right_on="ds",
                  how="left").drop(columns=["ds"])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model_ae(X_tensor)
        df["autoencoder_score"] = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    df["final_score"] = (
        0.3 * norm(df["iso_score"]) +
        0.2 * norm(df["kmeans_distance"]) +
        0.15 * norm(df["prophet_score"]) +
        0.15 * norm(df["autoencoder_score"]) +
        0.2 * norm(df["rule_score"])
    )

    threshold = df["final_score"].quantile(0.95)
    df["is_anomaly"] = df["final_score"] > threshold
    
    return df
