#distance_km (Input): How far the trip is, from 1.5 km to 25.0 km.
#hour_of_day (Input): What hour the trip starts, from 0 (midnight) to 23 (11 PM).
#is_peak (Input): Rush-hour flag—1 during morning (8, 9, 10) or evening (17, 18, 19, 20) peak hours, and 0 otherwise.
#traffic_index (Input): Road congestion level—1.0 means smooth/light traffic, and 2.5 means heavy traffic.
#mode (Input): Transport type—Transit, Bike, Auto, or Cab.
#fare (Target): The trip cost in ₹ that the model will learn to predict.


import numpy as np
import pandas as pd

# 1. Set a fixed seed so the random data is repeatable every time
rng = np.random.default_rng(42)
n_rows = 1500

# 2. Generate input features
distance_km = np.round(rng.uniform(1.5, 25.0, n_rows), 2)
hour_of_day = rng.integers(0, 24, n_rows)
is_peak = np.isin(hour_of_day, [8, 9, 10, 17, 18, 19, 20]).astype(int)
traffic_index = np.round(
    np.clip(1.0 + 0.45 * is_peak + rng.normal(0.15, 0.2, n_rows), 1.0, 2.5), 2
)
mode = rng.choice(
    ["Transit", "Bike", "Auto", "Cab"], size=n_rows, p=[0.2, 0.25, 0.25, 0.3]
)

# 3. Define realistic pricing rules per mode to compute target 'fare' (₹)
base_fare = {"Transit": 10, "Bike": 15, "Auto": 25, "Cab": 45}
per_km = {"Transit": 2.2, "Bike": 4.5, "Auto": 9.5, "Cab": 14.0}
peak_mult = {"Transit": 1.0, "Bike": 1.1, "Auto": 1.2, "Cab": 1.35}

fare = np.array([
    (
        base_fare[m]
        + per_km[m] * d
        * (1 + 0.25 * (t - 1.0))
        * (peak_mult[m] if p == 1 else 1.0)
    )
    for m, d, t, p in zip(mode, distance_km, traffic_index, is_peak)
])

# Add small real-world noise and keep minimum fare at ₹10
fare = np.round(np.maximum(fare + rng.normal(0, 5.0, n_rows), 10.0), 2)

# 4. Combine into a DataFrame and save to CSV
df = pd.DataFrame({
    "distance_km": distance_km,
    "hour_of_day": hour_of_day,
    "is_peak": is_peak,
    "traffic_index": traffic_index,
    "mode": mode,
    "fare": fare,
})

df.to_csv("commute_data.csv", index=False)

# 5. Inspect the dataset to verify there are no missing or broken values
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum())
print("\nSummary statistics:")
print(df.describe().round(2))