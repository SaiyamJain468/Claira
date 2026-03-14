import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from pathlib import Path
from typing import Union, Any

def load_prediction_assets(model_path: Path, scaler_path: Path):
    """
    Loads moving model and scaler assets.
    
    Args:
        model_path (Path): Path to the saved LightGBM model.
        scaler_path (Path): Path to the saved StandardScaler.
        
    Returns:
        tuple: (model, scaler)
    """
    model = lgb.Booster(model_file=str(model_path))
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_pm25(model: lgb.Booster, scaler: Any, features_df: pd.DataFrame) -> np.ndarray:
    """
    Predicts PM2.5 levels for a given set of features.
    
    Args:
        model (lgb.Booster): The trained LightGBM model.
        scaler (any): The fitted Standard scaler.
        features_df (pd.DataFrame): Input features for prediction.
        
    Returns:
        np.ndarray: Predicted PM2.5 values.
    """
    X_scaled = scaler.transform(features_df)
    return model.predict(X_scaled)
