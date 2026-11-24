# Rainfall Prediction System

A beginner-friendly machine learning minor project that predicts whether rainfall is likely using humidity, temperature, atmospheric pressure, and wind speed.

## Algorithms
- Decision Tree Classifier
- Random Forest Classifier

## Project structure

```
Rainfall-Prediction-System/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── train_model.py
    └── predict.py
```

## 1. Set up the environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Add your dataset

Save your CSV file as `data/rainfall.csv`. It must contain the columns:

`humidity, temperature, pressure, wind_speed, rainfall`

The first four columns are model inputs. The `rainfall` column is the answer the model learns to predict. See [data/README.md](data/README.md).

## 3. Train both models

From the project root folder, run:

```powershell
python src/train_model.py
```

The script prints accuracy, a classification report, and a confusion matrix for each model. It saves the trained models in the `models/` folder.

## 4. Run the prediction app

```powershell
streamlit run app.py
```

## 5. Make a prediction in the terminal

After training, run:

```powershell
python src/predict.py --humidity 70 --temperature 24 --pressure 1008 --wind-speed 12
```

## Important notes
- Use a dataset with correctly labelled rainfall values and enough examples of both classes.
- Model accuracy depends on dataset quality and class balance; do not claim a result before running the training script.
- The pressure and wind-speed units must match the units used in your dataset.
- The app gives an educational model estimate, not an official weather forecast.
