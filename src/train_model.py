"""Train Decision Tree and Random Forest rainfall classifiers."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "rainfall.csv"
MODEL_DIR = ROOT / "models"
FEATURES = ["humidity", "temperature", "pressure", "wind_speed"]
TARGET = "rainfall"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put your CSV file at data/rainfall.csv. See data/README.md."
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = [str(col).strip().lower() for col in data.columns]
    missing = [col for col in FEATURES + [TARGET] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    data = data[FEATURES + [TARGET]].dropna(subset=[TARGET])
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_text = data[TARGET].astype(str).str.strip().str.lower()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    if len(encoder.classes_) < 2:
        raise ValueError("The rainfall target must contain at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), FEATURES)]
    )

    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    for name, classifier in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        print(f"\n{name.replace('_', ' ').title()}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(classification_report(
            y_test, predictions, target_names=encoder.classes_, zero_division=0
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        joblib.dump(
            {"model": pipeline, "label_encoder": encoder, "features": FEATURES},
            MODEL_DIR / f"{name}.joblib"
        )
        print(f"Saved: models/{name}.joblib")


if __name__ == "__main__":
    main()
