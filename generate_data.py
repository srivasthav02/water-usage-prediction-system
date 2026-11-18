"""Generates simulated hourly household water usage (litres) with leaks injected.
Replace with real readings from the ESP32 flow sensor (see esp32_sensor/)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
hours = pd.date_range("2025-01-01", periods=24 * 90, freq="h")

base = np.array([2, 1, 1, 1, 3, 12, 25, 30, 15, 10, 8, 10,
                 15, 12, 8, 8, 10, 15, 22, 25, 18, 12, 6, 3], dtype=float)
usage = np.array([base[h.hour] * (1.3 if h.dayofweek >= 5 else 1.0) for h in hours])
usage += rng.normal(0, 1.5, len(hours))
usage = np.clip(usage, 0, None)

wastage = np.zeros(len(hours), dtype=int)
leak_idx = rng.choice(len(hours), size=40, replace=False)
usage[leak_idx] += rng.uniform(15, 40, size=40)      # simulated leak / tap left open
wastage[leak_idx] = 1

df = pd.DataFrame({"timestamp": hours, "litres": usage.round(2), "is_wastage": wastage})
df.to_csv("water_usage.csv", index=False)
print("Saved water_usage.csv", df.shape)
