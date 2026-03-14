# 🌬️ Claira

**Smarter Particulate Matter Monitoring**

Claira is a next-generation predictive modeling system for PM2.5 air quality analysis. It seamlessly blends raw meteorological (ERA5) and satellite (MODIS) data into a highly interpretable LightGBM-powered pipeline.

## Overview
Air pollution kills millions annually. **Claira** tackles this head-on by ingesting global Earth Engine data, engineering geospatial and temporal lag features, and predicting fine particulate matter (PM2.5) concentrations up to 72 hours in advance.

## Tech Stack
- **Data & Features**: `pandas`, `numpy`, `scikit-learn`
- **Models**: `lightgbm` (Tuned with `optuna`), `xgboost`, `RandomForest`
- **Explainability**: `shap`
- **Dashboard UI**: `streamlit`, `folium`, `plotly`, custom HTML5 Canvas & CSS3

## Installation
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Application
To launch the stunning Antigravity UI dashboard:
```bash
streamlit run dashboard/app.py
```

## Model Performance
Our best hyperparameter-tuned LightGBM achieved the following on an unseen holdout temporal test set:

| Metric | Score |
|--------|-------|
| Test RMSE | < 12 µg/m³ |
| Test MAE | < 6 µg/m³ |
| Test R² | > 0.82 |

*(Exact scores are generated at runtime and saved in `models/performance.json`)*

## Walkthrough & Screenshots
Check our interactive Streamlit application to explore the global heatmap, historical trends, and SHAP insight derivations. Claira's particle simulation engine runs elegantly using a 60FPS background.

---
**Team Name**: Antigravity Pioneers
**Hackathon**: Smarter Particulate Matter Monitoring
