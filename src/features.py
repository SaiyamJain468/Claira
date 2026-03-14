import pandas as pd
import numpy as np
import json
from pathlib import Path

def generate_features():
    proc_dir = Path("data/processed")
    src_dir = Path("src")
    df = pd.read_csv(proc_dir / "claira_clean.csv")
    
    # 3.1 Time Features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 3.2 Lag Features
    df = df.sort_values(by=['lat', 'lon', 'date'])
    df['pm25_lag1'] = df.groupby(['lat', 'lon'])['pm25'].shift(1)
    df['pm25_lag2'] = df.groupby(['lat', 'lon'])['pm25'].shift(2)
    df['pm25_lag3'] = df.groupby(['lat', 'lon'])['pm25'].shift(3)
    df['pm25_rolling3'] = df.groupby(['lat', 'lon'])['pm25'].transform(lambda x: x.rolling(3).mean())
    df['pm25_rolling6'] = df.groupby(['lat', 'lon'])['pm25'].transform(lambda x: x.rolling(6).mean())
    df = df.dropna(subset=['pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_rolling3', 'pm25_rolling6'])
    
    # 3.3 Meteorological Features
    df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    df['wind_direction'] = np.arctan2(df['v_component_of_wind_10m'], df['u_component_of_wind_10m']) * (180/np.pi)
    df['temp_humidity_index'] = df['temperature_2m'] * df['specific_humidity']
    df['pressure_hpa'] = df['surface_pressure'] / 100
    df['temp_celsius'] = df['temperature_2m'] - 273.15
    
    # 3.4 Satellite Features
    df['aod_mean'] = (df['Optical_Depth_047'] + df['Optical_Depth_055']) / 2
    df['aod_cloud_interaction'] = df['aod_mean'] * df['cloud_fraction']
    
    # 3.5 Geographic Features
    df['is_urban'] = (df['land_cover'] == 0).astype(int)
    
    # 3.6 Final Feature Set
    features = [
        'lat', 'lon', 'elevation', 'year', 'month_sin', 'month_cos', 'season',
        'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_rolling3', 'pm25_rolling6',
        'wind_speed', 'wind_direction', 'temp_humidity_index', 'pressure_hpa',
        'temp_celsius', 'aod_mean', 'aod_cloud_interaction', 'is_urban'
    ]
    target = 'pm25'
    
    print("Final features:", features)
    with open(src_dir / 'feature_list.json', 'w') as f:
        json.dump(features, f, indent=4)
        
    df.to_csv(proc_dir / "claira_features.csv", index=False)
    print(f"Feature engineering completed. Shape: {df.shape}")

if __name__ == "__main__":
    generate_features()
