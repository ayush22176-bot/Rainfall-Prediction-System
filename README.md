# Rainfall Prediction System

Predict whether rainfall is likely from **humidity**, **temperature**, **pressure**, and **wind speed**. This beginner-friendly project trains and compares a Decision Tree and a Random Forest classifier, then saves the stronger model.

## 1. 📌 Project Overview

Weather conditions can help estimate whether rainfall is likely. This project accepts four measurable weather features and returns a `Yes` or `No` rainfall prediction, alongside an estimated probability.

## 2. 🎯 Objectives

- Prepare weather data for binary rainfall classification.
- Train Decision Tree and Random Forest classifiers.
- Compare the models with accuracy, precision, recall, and F1-score.
- Save the best model and use it through a simple terminal interface.

## 3. 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Load and prepare CSV data |
| scikit-learn | Train and evaluate classifiers |
| joblib | Save and load the trained model |

## 4. 📊 Dataset Description

The required columns are shown below.

| Column | Meaning | Example |
|---|---|---|
| `humidity` | Relative humidity (%) | `78.5` |
| `temperature` | Air temperature (°C) | `26.0` |
| `pressure` | Atmospheric pressure (hPa) | `1004.2` |
| `wind_speed` | Wind speed (km/h) | `18.0` |
| `rainfall` | Target label (`Yes` / `No`) | `Yes` |

No external real-world dataset is bundled. On its first run, the training script creates a reproducible **synthetic demo dataset** in `data/rainfall.csv` so that the project works immediately. It is for learning only, not a real weather forecast. See [data/README.md](data/README.md) to replace it with a legitimate dataset and retrain.

## 5. ⚙️ System Implementation

1. Load `data/rainfall.csv` and verify its columns.
2. Split the data into 75% training and 25% testing data.
3. Train both classifiers with the same split.
4. Select the model with the highest rainfall-class F1-score.
5. Save it as `rainfall_model.pkl`; save the full comparison to `model_info.json`.

## 6. 🤖 Model Training

```bash
python train_model.py
```

The script reports a classification report for each model, then writes the selected model. With a replacement dataset, run this command again whenever the data changes.

## 7. 📈 Evaluation Metrics

| Evaluation Metric | Meaning |
|---|---|
| Accuracy | Overall share of correct predictions |
| Precision | How often predicted rainfall was actually rainfall |
| Recall | How much actual rainfall the model detected |
| F1-score | Balance between precision and recall |

Recorded results are generated on your computer during training and saved in `model_info.json`. This prevents misleading fixed metrics for a dataset that may be replaced.

## 8. 📁 Project Structure

```text
Rainfall-Prediction-System/
│
├── data/
│   └── README.md                 # Dataset format and replacement instructions
├── app.py                        # Interactive prediction interface
├── train_model.py                # Dataset creation, training, and comparison
├── rainfall_model.pkl            # Created after training; saved best model
├── model_info.json               # Created after training; evaluation results
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. ⚙️ Installation and Execution

**Step 1: Open a terminal in the project folder**

```bash
cd Rainfall-Prediction-System
```

**Step 2: Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**Step 3: Install the required packages**

```bash
python -m pip install -r requirements.txt
```

**Step 4: Train the models**

```bash
python train_model.py
```

**Step 5: Run a prediction**

```bash
python app.py
```

Enter the four requested weather values. The app will display whether rainfall is likely and its probability estimate.

## 10. ✅ Results and Conclusion

The project compares two common classification algorithms on the same held-out test data. The model with the better F1-score for rainfall is automatically selected. Random Forest often generalizes more reliably than one Decision Tree, but the saved choice should be based on the measured metrics for your dataset.

> **Important:** The included automatically generated data is educational. A reliable forecast system needs historical weather observations, careful validation across time and locations, and ongoing monitoring.
