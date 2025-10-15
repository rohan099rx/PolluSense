# 🌍 PolluSense — Air Quality Forecasting System# 🌍 PolluSense — Air Quality Forecasting System# 🌍 PolluSense — Air Quality Forecasting System# PolluSense — Short-term O3 / NO2 Forecasting Pipeline

**24/48-hour O₃ and NO₂ hourly forecasting for Delhi**

A production-ready machine learning system for predicting air pollutant concentrations using satellite data, meteorological forecasts, and historical ground observations.**24/48-hour O₃ and NO₂ hourly forecasting for Delhi**

---

## ✨ FeaturesA production-ready machine learning system for predicting air pollutant concentrations using satellite data, meteorological forecasts, and historical ground observations.**24/48-hour O₃ and NO₂ hourly forecasting for Delhi**This repository contains a starter pipeline to preprocess satellite and meteorological data, train a short-term forecasting model (24/48h hourly) and evaluate forecasts against ground observations. It's intended as a reproducible starting point for the SIH-2025 project.

- 🎯 **24/48-hour forecasts** with hourly resolution

- 📊 **Interactive dashboard** with Streamlit

- 🧠 **LightGBM models** with autoregressive features---

- 📈 **SHAP analysis** for model interpretability

- 🔧 **Clean data pipeline** with gap-filling and validation

- ⚡ **World-class accuracy** (R² = 0.977 for O3, 0.942 for NO2)

## ✨ FeaturesA production-ready machine learning system for predicting air pollutant concentrations using satellite data, meteorological forecasts, and historical ground observations.Key pieces:

---

## 🚀 Quick Start

- 🎯 **24/48-hour forecasts** with hourly resolution

### 1. Setup Environment

- 📊 **Interactive dashboard** with Streamlit

```bash

# Navigate to project- 🧠 **LightGBM models** with autoregressive features---- `src/preprocess.py` — ingestion, spatial/temporal harmonization (placeholder hooks) and feature engineering (lags, rolling stats).

cd PolluSense

- 📈 **SHAP analysis** for model interpretability

# Create virtual environment

python3 -m venv .venv- 🔧 **Clean data pipeline** with gap-filling and validation- `src/train.py` — dataset builder and LightGBM training + evaluation (RMSE, MAE, Bias).

source .venv/bin/activate  # On Windows: .venv\Scripts\activate

- ⚡ **World-class accuracy** (R² > 0.97 for O3)

# Install dependencies

pip install -r requirements.txt## ✨ Features- `src/predict.py` — load model and run forecast for a location/time range.

```

---

### 2. Run the Dashboard

- `config.yaml` — example configuration for file paths and model settings.

```bash

# Using the convenience script## 🚀 Quick Start

./start_dashboard.sh

- 🎯 **24/48-hour forecasts** with hourly resolution

# Or manually

streamlit run web/app.py### 1. Setup Environment

```

- 📊 **Interactive dashboard** with StreamlitData location (example): `/Users/admin/Downloads/Data_SIH_2025`.

The dashboard will open at **http://localhost:8501**

```bash

### 3. Use the Interface

# Clone and navigate to project- 🧠 **LightGBM models** with autoregressive features

- Select the **site_1_cleaned** model (recommended)

- Choose forecast horizon (24 or 48 hours)cd PolluSense

- View predictions with confidence intervals

- Analyze feature importance and SHAP values- 📈 **SHAP analysis** for model interpretabilityQuickstart

- Download forecasts as CSV

# Create virtual environment

---

python3 -m venv .venv- 🔧 **Clean data pipeline** with gap-filling and validation

## 📁 Project Structure

source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```

PolluSense/- ⚡ **7-8% better accuracy** with cleaned dataset1. Create a virtualenv and install dependencies:

├── README.md # This file

├── requirements.txt # Python dependencies# Install dependencies

├── config.yaml # Configuration

├── site_1_train_data_cleaned.csv # Training data (29,797 continuous hours)pip install -r requirements.txt

├── site_1_unseen_input_data.csv # Unseen test data

├── start_dashboard.sh # Dashboard launcher```

├── src/

│ ├── site_pipeline.py # Training pipeline with feature engineering---```bash

│ └── predict.py # Forecasting with recursive algorithm

├── web/### 2. Run the Dashboard

│ └── app.py # Streamlit dashboard

└── models/python3 -m venv .venv

    └── site_1_cleaned/               # Trained models (best quality)

        ├── site_models.joblib        # Model bundle```bash

        └── metrics.json              # Performance metrics

````# Using the convenience script## 🚀 Quick Startsource .venv/bin/activate



---./start_dashboard.sh



## 🎓 How It Workspip install -r requirements.txt



### Data Pipeline# Or manually

1. **Input**: Satellite data, meteorological forecasts, historical pollutant observations

2. **Preprocessing**: streamlit run web/app.py### 1. Setup Environment```

   - Gap-filling with time-aware interpolation

   - Feature engineering (lag features, rolling statistics)```

   - Temporal validation (train/test split)

3. **Training**: Per-pollutant LightGBM models with hyperparameter tuning

4. **Forecasting**: Recursive multi-step predictions with updated lag features

The dashboard will open at **http://localhost:8501**

### Key Features Used

- **Lag features**: O3/NO2 values at 1-48 hours ago```bash2. Preprocess data (adjust paths in `config.yaml`):

- **Rolling statistics**: 24-hour and 7-day moving averages

- **Meteorological forecasts**: Temperature, humidity, wind, pressure### 3. Generate Forecasts

- **Satellite data**: O3 and NO2 column forecasts

# Clone and navigate to project

### Model Performance

Use the Streamlit interface to:

**Test Set Performance (20% holdout - 5,955 samples):**

| Pollutant | RMSE  | MAE   | R²    | Correlation | Status |- Select the **site_1_cleaned** model (recommended)cd PolluSense```bash

|-----------|-------|-------|-------|-------------|--------|

| O₃        | 3.96  | 1.55  | 0.977 | 0.988       | ✅ Excellent |- Choose forecast horizon (24 or 48 hours)

| NO₂       | 6.49  | 3.77  | 0.942 | 0.971       | ✅ Excellent |

- View predictions with confidence intervalspython src/preprocess.py --config config.yaml --out processed_data.parquet

**⚠️ Important**: Model requires 48 hours of actual O3/NO2 measurements for accurate predictions. Without this historical data, predictions will be unreliable.

- Analyze feature importance and SHAP values

---

- Download forecasts as CSV# Create virtual environment```

## 🔧 Advanced Usage



### Retrain Models

---python3 -m venv .venv

```python

from src.site_pipeline import main as train_pipeline



# Train with default config## 📁 Project Structuresource .venv/bin/activate  # On Windows: .venv\Scripts\activate3. Train model:

train_pipeline('config.yaml', 'site_1_train_data_cleaned.csv', 'models/my_site/')

````

### Programmatic Forecasting```

``````````pythonPolluSense/

import joblib

import pandas as pd├── README.md                          # This file# Install dependencies```bash

from src.predict import recursive_forecast

from src.site_pipeline import add_past_target_features├── requirements.txt                   # Python dependencies



# Load model and history├── config.yaml                        # Configurationpip install -r requirements.txtpython src/train.py --config config.yaml --input processed_data.parquet --out models/

models = joblib.load('models/site_1_cleaned/site_models.joblib')

history = pd.read_csv('site_1_train_data_cleaned.csv')├── site_1_train_data_cleaned.csv     # Training data (29,797 continuous hours)



# Prepare history with lag features├── site_1_unseen_input_data.csv      # Unseen test data``````

history['datetime'] = pd.to_datetime(history[['year', 'month', 'day', 'hour']])

history = history.set_index('datetime').sort_index()├── evaluate_unseen_data.py           # Model evaluation script

history = add_past_target_features(history, ['O3_target', 'NO2_target'])

├── start_dashboard.sh                # Dashboard launcher

# Generate 24-hour forecast

predictions = recursive_forecast(├── src/

    models,

    history.iloc[-48:],  # Use last 48 hours│   ├── site_pipeline.py              # Training pipeline with feature engineering### 2. Run the Dashboard4. Predict:

    steps=24,

    features=models['features']│   └── predict.py                    # Forecasting with recursive algorithm

)

├── web/

# Extract results

for timestamp, values in predictions:│   └── app.py                        # Streamlit dashboard

    print(f"{timestamp}: O3={values['O3_target']:.2f}, NO2={values['NO2_target']:.2f}")

```└── models/```bash```bash



---    └── site_1_cleaned/               # Trained models (best quality)



## 📊 Data Quality        ├── site_models.joblib        # Model bundlestreamlit run web/app.pypython src/predict.py --config config.yaml --model models/lightgbm_o3_no2.pkl --start 2025-01-01T00:00 --steps 24



The cleaned dataset (`site_1_train_data_cleaned.csv`) addresses critical issues:        └── metrics.json              # Performance metrics



### Problems Fixed`````````

- ❌ **393 time series gaps** (42% missing data) → ✅ **Continuous hourly series**

- ❌ **96% missing satellite columns** → ✅ **Dropped sparse columns**

- ❌ **96.8% data loss after dropna** → ✅ **99.9% retention with interpolation**

---

### Improvements

- 📈 **24% more training data** (29,797 vs 25,081 hours)

- 📉 **Significantly better performance** (R² = 0.977 for O3)

- 🎯 **Lower error rates** (MAE = 1.55 µg/m³ for O3)## 🎓 How It WorksThe dashboard will open at **http://localhost:8501**Notes



---



## 🛠 Technology Stack### Data Pipeline



- **Core**: Python 3.12, pandas, numpy1. **Input**: Satellite data, meteorological forecasts, historical pollutant observations

- **ML**: LightGBM (gradient boosting)

- **Visualization**: Streamlit, matplotlib, SHAP2. **Preprocessing**: ### 3. Generate Forecasts- The preprocessing stage includes hooks for spatial regridding and satellite/ground co-location. For a first run, ensure your data are hourly and in CSV/Parquet format or adjust `src/preprocess.py` to load your file types.

- **Data**: CSV, joblib (model serialization)

   - Gap-filling with time-aware interpolation

---

   - Feature engineering (lag features, rolling statistics)- This starter uses LightGBM for speed and interpretability. For capturing complex spatio-temporal dependencies, consider a convolutional LSTM or Transformer architecture in a follow-up.

## 🐛 Troubleshooting

   - Temporal validation (train/test split)

### Predictions are unreliable

✅ **Solution**: Ensure you provide 48 hours of actual O3/NO2 measurements as history. The model uses lagged features (past 1-48 hours) which are critical for accurate predictions.3. **Training**: Per-pollutant LightGBM models with hyperparameter tuningUse the Streamlit interface to:



### Model performance is poor4. **Forecasting**: Recursive multi-step predictions with updated lag features

- Ensure you're using `models/site_1_cleaned/` (not old versions)

- Use `site_1_train_data_cleaned.csv` for history- Select the **site_1_cleaned** model (recommended)License: MIT

- Check that history has at least 48 hours of continuous data

- Verify no gaps in the historical measurements### Key Features Used



### Dashboard won't load- **Lag features**: O3/NO2 values at 1-48 hours ago- Choose forecast horizon (24 or 48 hours)

```bash

# Check if Streamlit is installed- **Rolling statistics**: 24-hour and 7-day moving averages- View predictions with confidence intervals

pip install streamlit

- **Meteorological forecasts**: Temperature, humidity, wind, pressure- Analyze feature importance and SHAP values

# Run with verbose logging

streamlit run web/app.py --logger.level=debug- **Satellite data**: O3 and NO2 column forecasts- Download forecasts as CSV

``````````

### SHAP plots not displaying

```bash### Model Performance (site_1_cleaned)---

# Update matplotlib

pip install --upgrade matplotlib shap

```

**With Proper Features (Complete 48-hour History):**## 📁 Project Structure

---

| Pollutant | RMSE | MAE | R² | Correlation | Status |

## 📝 Important Notes

|-----------|-------|-------|-------|-------------|--------|```

### About Model Performance

The model achieves **excellent performance** (R² = 0.977 for O3, 0.942 for NO2) on the test set with:| O₃ | 7.67 | 4.79 | 0.971 | 0.985 | ✅ Excellent |PolluSense/

- 48 hours of actual pollutant measurements (for lag features)

- Meteorological forecasts| NO₂ | 9.49 | 5.81 | 0.893 | 0.946 | ✅ Excellent |├── README.md # This file

- Satellite observations

├── QUICKSTART.md # Quick start guide

See `MODEL_VALIDATION_SUMMARY.md` for detailed validation analysis.

**⚠️ Important**: Model requires 48 hours of actual O3/NO2 measurements for accurate predictions. Without this historical data, predictions will be unreliable.├── requirements.txt # Python dependencies

### Production Deployment Requirements

For production use, you need:├── config.yaml # Configuration

1. **Real-time data feed** providing past 48 hours of O3/NO2 measurements

2. **Continuous monitoring** to detect data gaps---├── site_1_train_data_cleaned.csv # Training data (29,797 continuous hours)

3. **Fallback strategy** for handling missing historical data

4. **Regular model retraining** with recent data├── src/

---## 🔧 Advanced Usage│ ├── site_pipeline.py # Training pipeline with feature engineering

## 📄 License│ └── predict.py # Forecasting with recursive algorithm

MIT License - Feel free to use and modify for your projects.### Evaluate on Unseen Data├── web/

---│ └── app.py # Streamlit dashboard

## 🙏 Acknowledgments```bash└── models/

- Data sources: Satellite observations, meteorological forecasts, CPCB ground measurements.venv/bin/python evaluate_unseen_data.py └── site_1_cleaned/ # Trained models (best quality)

- Built for SIH-2025 air quality prediction challenge

- Model achieves state-of-the-art performance for Delhi region``` ├── site_models.joblib # Model bundle

--- └── metrics.json # Performance metrics

**Need help?** Check `MODEL_VALIDATION_SUMMARY.md` or open an issue on GitHub.This will:```

- Load the unseen test data

- Generate predictions---

- Create visualization plots

- Save results to `site_1_unseen_predictions.csv`## 🎓 How It Works

### Retrain Models### Data Pipeline

1. **Input**: Satellite data, meteorological forecasts, historical pollutant observations

````python2. **Preprocessing**:

from src.site_pipeline import main as train_pipeline   - Gap-filling with time-aware interpolation

   - Feature engineering (lag features, rolling statistics)

# Train with default config   - Temporal validation (train/test split)

train_pipeline('config.yaml', 'site_1_train_data_cleaned.csv', 'models/my_site/')3. **Training**: Per-pollutant LightGBM models with hyperparameter tuning

```4. **Forecasting**: Recursive multi-step predictions with updated lag features



### Programmatic Forecasting### Key Features Used

- **Lag features**: O3/NO2 values at 1h, 2h, 3h, 6h, 12h, 24h ago

```python- **Rolling statistics**: 24-hour and 7-day moving averages

import joblib- **Meteorological forecasts**: Temperature, humidity, wind, pressure

import pandas as pd- **Satellite data**: O3 and NO2 column forecasts

from src.predict import recursive_forecast

from src.site_pipeline import add_past_target_features### Model Performance (site_1_cleaned)

| Pollutant | RMSE  | MAE   | R²    | Improvement |

# Load model and history|-----------|-------|-------|-------|-------------|

models = joblib.load('models/site_1_cleaned/site_models.joblib')| O₃        | 8.86  | 6.12  | 0.959 | 7.5% better |

history = pd.read_csv('site_1_train_data_cleaned.csv')| NO₂       | 8.11  | 5.71  | 0.901 | 7.0% better |



# Prepare history with lag features---

history['datetime'] = pd.to_datetime(history[['year', 'month', 'day', 'hour']])

history = history.set_index('datetime').sort_index()## 🔧 Advanced Usage

history = add_past_target_features(history, ['O3_target', 'NO2_target'])

### Retrain Models

# Generate 24-hour forecast

predictions = recursive_forecast(```python

    models, from src.site_pipeline import main as train_pipeline

    history.iloc[-48:],  # Use last 48 hours

    steps=24, # Train with default config

    features=models['features']train_pipeline('config.yaml', 'site_1_train_data_cleaned.csv', 'models/my_site/')

)```



# Extract results### Programmatic Forecasting

for timestamp, values in predictions:

    print(f"{timestamp}: O3={values['O3_target']:.2f}, NO2={values['NO2_target']:.2f}")```python

```import joblib

import pandas as pd

---from src.predict import recursive_forecast

from src.site_pipeline import add_past_target_features

## 📊 Data Quality

# Load model and history

The cleaned dataset (`site_1_train_data_cleaned.csv`) addresses critical issues:models = joblib.load('models/site_1_cleaned/site_models.joblib')

history = pd.read_csv('site_1_train_data_cleaned.csv')

### Problems Fixed

- ❌ **393 time series gaps** (42% missing data) → ✅ **Continuous hourly series**# Prepare history with lag features

- ❌ **96% missing satellite columns** → ✅ **Dropped sparse columns**history['datetime'] = pd.to_datetime(history[['year', 'month', 'day', 'hour']])

- ❌ **96.8% data loss after dropna** → ✅ **99.9% retention with interpolation**history = history.set_index('datetime').sort_index()

history = add_past_target_features(history, ['O3_target', 'NO2_target'])

### Improvements

- 📈 **24% more training data** (29,797 vs 25,081 hours)# Generate 24-hour forecast

- 📉 **Better RMSE** across all pollutantspredictions = recursive_forecast(

- 🎯 **Higher R² scores** (O3: 0.971, NO2: 0.893)    models,

    history.iloc[-48:],  # Use last 48 hours

---    steps=24,

    features=models['features']

## 🛠 Technology Stack)



- **Core**: Python 3.12, pandas, numpy# Extract results

- **ML**: LightGBM (gradient boosting)for timestamp, values in predictions:

- **Visualization**: Streamlit, matplotlib, SHAP    print(f"{timestamp}: O3={values['O3_target']:.2f}, NO2={values['NO2_target']:.2f}")

- **Data**: CSV/Parquet, joblib (model serialization)```



------



## 🐛 Troubleshooting## 📊 Data Quality



### Predictions are unreliableThe cleaned dataset (`site_1_train_data_cleaned.csv`) addresses critical issues:

✅ **Solution**: Ensure you provide 48 hours of actual O3/NO2 measurements as history. The model uses lagged features (past 1-48 hours) which are critical for accurate predictions.

### Problems Fixed

### Model performance is poor- ❌ **393 time series gaps** (42% missing data) → ✅ **Continuous hourly series**

- Ensure you're using `models/site_1_cleaned/` (not old versions)- ❌ **96% missing satellite columns** → ✅ **Dropped sparse columns**

- Use `site_1_train_data_cleaned.csv` for history- ❌ **96.8% data loss after dropna** → ✅ **99.9% retention with interpolation**

- Check that history has at least 48 hours of continuous data

- Verify no gaps in the historical measurements### Improvements

- 📈 **24% more training data** (29,797 vs 25,081 hours)

### Dashboard won't load- 📉 **7-8% better RMSE** across all pollutants

```bash- 🎯 **Higher R² scores** (O3: 0.959, NO2: 0.901)

# Check if Streamlit is installed

pip install streamlitSee `QUICKSTART.md` for step-by-step instructions.



# Run with verbose logging---

streamlit run web/app.py --logger.level=debug

```## 🛠 Technology Stack



### SHAP plots not displaying- **Core**: Python 3.12, pandas, numpy

This was fixed in the latest version. If you still see issues:- **ML**: LightGBM (gradient boosting)

```bash- **Visualization**: Streamlit, matplotlib, SHAP

# Update matplotlib- **Data**: CSV/Parquet, joblib (model serialization)

pip install --upgrade matplotlib shap

```---



---## 🐛 Troubleshooting



## 📝 Important Notes### Predictions are stuck at same values

✅ **Fixed!** The recursive forecast now properly updates lag features after each step.

### About Model Validation

The model achieves **excellent performance** (R² = 0.971 for O3, 0.893 for NO2) when evaluated with complete data that includes:### Model performance is poor

- 48 hours of actual pollutant measurements (for lag features)- Ensure you're using `models/site_1_cleaned/` (not the old `site_1`)

- Meteorological forecasts- Use `site_1_train_data_cleaned.csv` for history

- Satellite observations- Check that history has at least 48 hours of data



See `MODEL_VALIDATION_SUMMARY.md` for detailed validation analysis.### Dashboard won't load

```bash

### Production Deployment Requirements# Check if Streamlit is installed

For production use, you need:pip install streamlit

1. **Real-time data feed** providing past 48 hours of O3/NO2 measurements

2. **Continuous monitoring** to detect data gaps# Run with verbose logging

3. **Fallback strategy** for handling missing historical datastreamlit run web/app.py --logger.level=debug

4. **Regular model retraining** with recent data```



------



## 📄 License## 📝 Citation



MIT License - Feel free to use and modify for your projects.If you use this system in research or production, please cite:



---```

PolluSense Air Quality Forecasting System

## 🙏 Acknowledgmentshttps://github.com/yourusername/PolluSense

````

- Data sources: Satellite observations, meteorological forecasts

- Built for SIH-2025 air quality prediction challenge---

- Model validation confirmed world-class performance

## 📄 License

---

MIT License - Feel free to use and modify for your projects.

**Need help?** Open an issue or check the documentation files in the repository.

---

## 🙏 Acknowledgments

- Data sources: Satellite observations, meteorological forecasts
- Built for SIH-2025 air quality prediction challenge
- Model improvements achieved through systematic data quality analysis

---

**Need help?** Check `QUICKSTART.md` for step-by-step instructions or open an issue on GitHub.
