# 🌬️ CLAIRA — Advanced PM2.5 Intelligence System

> **Smarter Particulate Matter Monitoring** | Hackathon Project by **Antigravity Pioneers**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?style=flat-square&logo=streamlit)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Overview

Air pollution kills **7 million people annually**. **CLAIRA** tackles this head-on by:
- Ingesting global **ERA5 meteorological** and **MODIS satellite** data via Google Earth Engine
- Engineering geospatial and temporal lag features
- Predicting **PM2.5 concentrations up to 72 hours** in advance using an Optuna-tuned LightGBM model
- Visualizing results through a real-time, animated Streamlit dashboard

---

## 🏗️ Project Structure

```
claira/
├── dashboard/          # Streamlit UI (app.py + assets)
├── data/               # Raw & processed datasets
├── models/             # Trained model files & performance metrics
├── notebooks/          # EDA & experimentation notebooks
├── reports/            # SHAP plots, insights & figures
├── src/                # Core pipeline: data loading, features, model, predict
├── requirements.txt    # Python dependencies
└── app.py              # Deployment entrypoint
```

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Data & Features** | `pandas`, `numpy`, `scikit-learn`, Google Earth Engine |
| **Modeling** | `lightgbm`, `xgboost`, `optuna`, `shap` |
| **Dashboard** | `streamlit`, `plotly`, `folium`, HTML5 Canvas, CSS3 |
| **Geospatial** | `folium`, `pydeck`, `xyzservices` |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SaiyamJain468/Claira.git
cd Claira
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

Best results from the Optuna-tuned LightGBM on a **temporal holdout test set**:

| Metric | Score |
|--------|-------|
| **Test RMSE** | < 12 µg/m³ |
| **Test MAE** | < 6 µg/m³ |
| **Test R²** | > 0.82 |

> Exact scores are generated at runtime and saved to `models/performance.json`

---

## ☁️ Deployment

This app is optimized for **[Streamlit Community Cloud](https://share.streamlit.io)**:

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app** → select `SaiyamJain468/Claira`
3. Set **Main file path** to `dashboard/app.py`
4. Click **Deploy**

---

## ✨ Features

- **⚡ Real-Time Monitor** — Live PM2.5 readings with animated particle simulation at 60FPS
- **🗺️ Global Risk Map** — Interactive Folium heatmap of worldwide particulate nodes
- **🔮 72-Hour Forecast** — Multi-step ahead predictions with trend indicators
- **🧠 SHAP Explainability** — Feature importance breakdown for every prediction
- **☣️ Health Guidelines** — Dynamic risk levels (Good → Danger) with adaptive UI theming

---

## 👥 Team

**Antigravity Pioneers** — Built for the *Smarter Particulate Matter Monitoring* Hackathon
