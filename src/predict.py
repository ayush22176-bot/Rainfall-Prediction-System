"""Make a rainfall prediction from weather measurements."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"


def predict(humidity, temperature, pressure, wind_speed):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train_model.py")
    saved = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "humidity": humidity,
        "temperature": temperature,
        "pressure": pressure,
        "wind_speed": wind_speed,
    }], columns=saved["features"])
    result = saved["model"].predict(row)[0]
    return saved["label_encoder"].inverse_transform([result])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether rainfall is likely.")
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    args = parser.parse_args()
    answer = predict(args.humidity, args.temperature, args.pressure, args.wind_speed)
    print(f"Rainfall prediction: {answer}")
