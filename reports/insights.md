# Claira PM2.5 Insights
    
1. **Critical Drivers (Insight 1)**: The single most impactful feature for predicting PM2.5 is **pm25_rolling3**. This dominates the model's decision tree splits, driving the largest magnitude of SHAP values globally.
2. **Meteorological Impact (Insight 2)**: Higher wind speeds generally disperse particulate matter. The SHAP dependence plots indicate a strong inverse relationship between wind speed and predicted PM2.5, driving predictions down by a significant marginal amount when wind speed exceeds the local average.
3. **Temporal vs Real-time (Insight 3)**: PM2.5 is heavily driven by local history. Lag features and rolling averages establish a strong baseline, while real-time meteorology and satellite AOD explain the acute daily variations.

*Generated automatically via SHAP explainability analysis.*
