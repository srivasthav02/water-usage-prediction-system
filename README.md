# Smart Water Usage Prediction and Wastage Alert System

> **Status: Prototype.** The ML pipeline runs on simulated data; the ESP32 sketch is the planned real data source.

## Overview
Predicts normal hourly water consumption and raises an alert when usage is far above what is expected (leaks, taps left open).

## Pipeline
1. `generate_data.py` creates 90 days of simulated hourly usage with injected leaks.
2. `train_predict.py` trains a Random Forest on normal usage and flags hours where actual usage exceeds prediction by more than 3 standard deviations.
3. `esp32_sensor/flow_sensor.ino` logs real flow-sensor readings (YF-S201).

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py
python train_predict.py
```
Output: MAE, alert counts, and `result.png`.

## Future Improvements
- Live ingestion over WiFi/MQTT, mobile/web notifications
- Per-household baselines

## Author
Beeravelli Srivasthav
