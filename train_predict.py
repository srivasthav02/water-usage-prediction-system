"""Train a usage predictor and flag wastage when actual >> predicted."""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("water_usage.csv", parse_dates=["timestamp"])
df["hour"] = df.timestamp.dt.hour
df["dow"] = df.timestamp.dt.dayofweek
df["lag24"] = df.litres.shift(24)
df = df.dropna().reset_index(drop=True)

features = ["hour", "dow", "lag24"]
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:].copy()

# Train on normal hours only so the model learns usual consumption
normal = train[train.is_wastage == 0]
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(normal[features], normal.litres)

test["predicted"] = model.predict(test[features])
print("MAE (litres):", round(mean_absolute_error(test.litres, test.predicted), 2))

# Wastage rule: actual exceeds prediction by > 3 standard deviations of normal residuals
resid = normal.litres - model.predict(normal[features])
threshold = 3 * resid.std()
test["alert"] = (test.litres - test.predicted > threshold).astype(int)

tp = ((test.alert == 1) & (test.is_wastage == 1)).sum()
fp = ((test.alert == 1) & (test.is_wastage == 0)).sum()
fn = ((test.alert == 0) & (test.is_wastage == 1)).sum()
print(f"Alerts: TP={tp} FP={fp} FN={fn}")

joblib.dump({"model": model, "threshold": threshold}, "model.joblib")

week = test.iloc[:24 * 7]
plt.figure(figsize=(11, 4))
plt.plot(week.timestamp, week.litres, label="Actual")
plt.plot(week.timestamp, week.predicted, label="Predicted")
a = week[week.alert == 1]
plt.scatter(a.timestamp, a.litres, color="red", label="Wastage alert", zorder=3)
plt.legend(); plt.ylabel("Litres/hour"); plt.tight_layout()
plt.savefig("result.png", dpi=150)
print("Saved result.png")
