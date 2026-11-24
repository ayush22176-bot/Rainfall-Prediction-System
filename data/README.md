# Dataset instructions

`train_model.py` automatically creates `rainfall.csv` the first time it runs. It is a **synthetic demonstration dataset** made from weather-like ranges, intended only to make the project runnable without a download.

For a real project, replace it with your own CSV named `rainfall.csv` in this folder. It must have these exact columns:

```text
humidity,temperature,pressure,wind_speed,rainfall
```

Use numbers for the first four columns. The `rainfall` value must be `Yes` or `No`. Use a trustworthy local weather-data source, document its origin and licensing, and retrain with `python train_model.py`.
