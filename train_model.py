"""Train and compare rainfall prediction classifiers.

Run: python train_model.py
The script creates a small, clearly labelled demo dataset if data/rainfall.csv
is absent. Replace it with a real dataset for a real-world project.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rainfall.csv"
MODEL_PATH = BASE_DIR / "rainfall_model.pkl"
INFO_PATH = BASE_DIR / "model_info.json"
FEATURES = ["humidity", "temperature", "pressure", "wind_speed"]
TARGET = "rainfall"
RANDOM_STATE = 42


def create_demo_dataset(path: Path, rows: int = 240) -> None:
    """Make a reproducible demo-only dataset with weather-like values."""
    random.seed(RANDOM_STATE)
    records = []
    for _ in range(rows):
        humidity = round(random.uniform(35, 100), 1)
        temperature = round(random.uniform(15, 42), 1)
        pressure = round(random.uniform(990, 1028), 1)
        wind_speed = round(random.uniform(0, 45), 1)
        rain_score = (humidity - 62) * 0.10 + (1012 - pressure) * 0.12 + wind_speed * 0.025 - (temperature - 25) * 0.03
        rainfall = "Yes" if rain_score + random.uniform(-1.2, 1.2) > 0 else "No"
        records.append({"humidity": humidity, "temperature": temperature, "pressure": pressure, "wind_speed": wind_speed, "rainfall": rainfall})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Created demo dataset: {path}")


def calculate_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label="Yes", average="binary", zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4)}


def main() -> None:
    if not DATA_PATH.exists():
        create_demo_dataset(DATA_PATH)

    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")
    if not set(data[TARGET].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("The rainfall column must contain only Yes and No.")

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=RANDOM_STATE),
    }
    results, trained_models = {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results[name] = calculate_metrics(y_test, prediction)
        trained_models[name] = model
        print(f"\n{name}")
        print(pd.DataFrame([results[name]]).to_string(index=False))
        print(classification_report(y_test, prediction, zero_division=0))

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    artifact = {"model": trained_models[best_name], "model_name": best_name, "features": FEATURES}
    joblib.dump(artifact, MODEL_PATH)
    INFO_PATH.write_text(json.dumps({"best_model": best_name, "metrics": results, "dataset_rows": len(data), "note": "Metrics use a 25% hold-out test split (random_state=42)."}, indent=2), encoding="utf-8")
    print(f"\nSaved best model ({best_name}) to {MODEL_PATH.name}")
    print(f"Saved comparison metrics to {INFO_PATH.name}")


if __name__ == "__main__":
    main()
