import pandas as pd
from prophet import Prophet
import numpy as np


class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, df, date_col="date", value_col="amount"):
        # Prophet requires specific column names
        prophet_df = df[[date_col, value_col]].rename(
            columns={date_col: "ds", value_col: "y"}
        )

        self.model.fit(prophet_df)
        self.history = prophet_df

    def predict(self):
        forecast = self.model.predict(self.history)

        # Calculate anomaly score based on deviation
        actual = self.history["y"].values
        predicted = forecast["yhat"].values

        error = np.abs(actual - predicted)

        # Normalize
        score = (error - error.min()) / (error.max() - error.min())

        return score