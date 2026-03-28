from utils.preprocessing import load_data, preprocess
from models.autoencoder import train_autoencoder, get_ae_scores
from models.clustering_iforest import ClusteredIForest
from models.prophet_model import ProphetModel
from models.ensemble import weighted_score

import pandas as pd


def run_training(data_path):
    df = load_data(data_path)
    X, scaler = preprocess(df)

    # -------------------
    # 1. Autoencoder
    # -------------------
    ae_model = train_autoencoder(X)
    ae_scores = get_ae_scores(ae_model, X)

    # -------------------
    # 2. Clustered IF
    # -------------------
    if_model = ClusteredIForest()
    if_model.fit(X)
    if_scores = if_model.score(X)

    # -------------------
    # 3. Prophet (NEW)
    # -------------------
    prophet = ProphetModel()
    prophet.fit(df, date_col="date", value_col="amount")
    prophet_scores = prophet.predict()

    # -------------------
    # 4. Rule-based
    # -------------------
    df["rule_flag"] = df["amount"] > 5000
    rule_score = df["rule_flag"].astype(int)

    # -------------------
    # 5. Ensemble (UPDATED)
    # -------------------
    final_score = (
        0.4 * ae_scores +
        0.3 * if_scores +
        0.2 * prophet_scores +
        0.1 * rule_score
    )

    df["anomaly_score"] = final_score
    df["is_anomaly"] = df["anomaly_score"] > 0.7

    df.to_csv("outputs/results.csv", index=False)

    print("Training completed with Prophet integration.")