# Let's load the uploaded dataset, expand ORDER_DATE from Jan 2025 to Mar 2026,
# generate >1,000,000 rows with day-by-day progression and random daily volumes,
# and save to CSV.

import pandas as pd
import numpy as np

file_path = "data/raw_dataset.csv"
df = pd.read_csv(file_path)

# Ensure ORDER_DATE exists or create if missing
if "ORDER_DATE" not in df.columns:
    df["ORDER_DATE"] = pd.to_datetime("2025-01-01")

# Convert to datetime
df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors='coerce')

# Define date range
date_range = pd.date_range(start="2025-01-01", end="2026-03-31", freq="D")

# Target total rows > 1,000,000
target_rows = 1_000_000

# Generate random daily counts that sum to target_rows
np.random.seed(42)
random_weights = np.random.rand(len(date_range))
daily_counts = (random_weights / random_weights.sum() * target_rows).astype(int)

# Fix rounding to ensure exact total
difference = target_rows - daily_counts.sum()
daily_counts[0] += difference

# Generate expanded dataset
expanded_rows = []

for date, count in zip(date_range, daily_counts):
    sampled = df.sample(n=count, replace=True).copy()
    sampled["ORDER_DATE"] = date
    expanded_rows.append(sampled)

expanded_df = pd.concat(expanded_rows, ignore_index=True)

# Save to CSV
output_path = "expanded_dataset.csv"
expanded_df.to_csv(output_path, index=False)

output_path