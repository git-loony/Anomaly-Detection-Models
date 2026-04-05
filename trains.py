import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder

# FEATURES
features = [
    "price", "quantity", "total_amount", "log_amount",
    "order_count", "avg_spend_customer", "std_spend_customer",
    "day_of_week", "is_weekend",
    "payment_method", "country", "city", "category"
]

# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):
    df = df.copy()

    df["total_amount"] = df["price"] * df["quantity"]
    df["log_amount"] = np.log1p(df["total_amount"])

    df["order_count"] = df.groupby("customer_id")["order_id"].transform("count")
    df["avg_spend_customer"] = df.groupby("customer_id")["total_amount"].transform("mean")
    df["std_spend_customer"] = df.groupby("customer_id")["total_amount"].transform("std").fillna(0)

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    cat_cols = ["payment_method", "country", "city", "category"]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders

# =========================
# RULES
# =========================
def rule_based_checks(df):
    df = df.copy()

    df["rule_duplicate"] = df.duplicated(
        subset=["customer_id", "product_id", "order_date", "total_amount"],
        keep=False
    ).astype(int)

    category_avg = df.groupby("category")["total_amount"].transform("mean")
    df["rule_price_spike"] = (df["total_amount"] > 3 * category_avg).astype(int)

    df["rule_quantity"] = (df["quantity"] > df["quantity"].quantile(0.99)).astype(int)

    payment_freq = df["payment_method"].value_counts(normalize=True)
    df["rule_payment"] = df["payment_method"].map(payment_freq) < 0.02
    df["rule_payment"] = df["rule_payment"].astype(int)

    customer_country = df.groupby("customer_id")["country"].transform(lambda x: x.mode()[0])
    df["rule_geo"] = (df["country"] != customer_country).astype(int)

    df["rule_score"] = (
        2 * df["rule_duplicate"] +
        2 * df["rule_price_spike"] +
        1 * df["rule_quantity"] +
        1 * df["rule_payment"] +
        1 * df["rule_geo"]
    )

    return df

# =========================
# AUTOENCODER
# =========================
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