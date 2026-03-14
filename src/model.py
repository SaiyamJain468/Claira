import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

def train_and_eval() -> None:
    """
    Trains and evaluates Random Forest, LightGBM, and XGBoost models.
    Performs hyperparameter optimization using Optuna for LightGBM.
    Saves the final model, scaler, and performance metrics.
    
    Returns:
        None
    """
    proc_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/figures")
    
    df = pd.read_csv(proc_dir / "claira_features.csv")
    with open("src/feature_list.json", "r") as f:
        features = json.load(f)
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['lat', 'lon', 'date']) # actually should just sort by date, wait: T084 says sort all data by date
    df = df.sort_values(by='date')
    
    # Split
    train_mask = df['date'] < '2021-10-01'
    val_mask = (df['date'] >= '2021-10-01') & (df['date'] < '2022-04-01')
    test_mask = df['date'] >= '2022-04-01'
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'pm25']
    X_val, y_val = df.loc[val_mask, features], df.loc[val_mask, 'pm25']
    X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, 'pm25']

    df.loc[test_mask, ['lat', 'lon', 'date', 'pm25'] + features].to_csv(proc_dir / "test_data_with_coords.csv", index=False)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    
    # Baseline RandomForest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_val_scaled)
    print("RF Validation - RMSE:", np.sqrt(mean_squared_error(y_val, rf_pred)), 
          "MAE:", mean_absolute_error(y_val, rf_pred), "R2:", r2_score(y_val, rf_pred))
    
    # LightGBM Custom
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
        'num_leaves': 64, 'max_depth': -1, 'min_data_in_leaf': 20,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
        'feature_pre_filter': False
    }
    evals_result = {}
    lgb_model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[train_data, val_data], 
                          callbacks=[lgb.early_stopping(50), lgb.record_evaluation(evals_result)])
                          
    lgb_pred = lgb_model.predict(X_val_scaled)
    print("LGBM Validation - RMSE:", np.sqrt(mean_squared_error(y_val, lgb_pred)),
          "MAE:", mean_absolute_error(y_val, lgb_pred), "R2:", r2_score(y_val, lgb_pred))
          
    # Plot RMSE curve
    lgb.plot_metric(evals_result, metric='rmse')
    plt.tight_layout()
    plt.savefig(report_dir / "lgbm_training_curve.png")
    plt.close()
    
    # XGBoost Default
    xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, early_stopping_rounds=50)
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_val_scaled)
    print("XGB Validation - RMSE:", np.sqrt(mean_squared_error(y_val, xgb_pred)))
    
    # Optuna tuning for LightGBM
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        param = {
            'objective': 'regression', 'metric': 'rmse', 'verbose': -1,
            'feature_pre_filter': False,
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100)
        }
        model = lgb.train(param, train_data, num_boost_round=500, valid_sets=[val_data], 
                          callbacks=[lgb.early_stopping(50)])
        preds = model.predict(X_val_scaled)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study = optuna.create_study(direction="minimize")
    print("Starting Optuna tuning (50 trials)...")
    study.optimize(objective, n_trials=50)
    print("Best params:", study.best_params)
    
    # Retrain on train+val
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = pd.concat([y_train, y_val])
    train_val_data = lgb.Dataset(X_train_val, label=y_train_val)
    
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbose': -1})
    final_model = lgb.train(best_params, train_val_data, num_boost_round=500)
    
    final_preds = final_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    test_mae = mean_absolute_error(y_test, final_preds)
    test_r2 = r2_score(y_test, final_preds)
    
    print(f"FINAL Test Evaluation - RMSE: {test_rmse}, MAE: {test_mae}, R2: {test_r2}")
    
    final_model.save_model(models_dir / "claira_lgbm.txt")
    
    perf = {
        "Val RMSE": float(np.sqrt(mean_squared_error(y_val, lgb_pred))),
        "Test RMSE": float(test_rmse),
        "Test MAE": float(test_mae),
        "Test R2": float(test_r2)
    }
    with open(models_dir / "performance.json", "w") as f:
        json.dump(perf, f)
        
if __name__ == "__main__":
    train_and_eval()
