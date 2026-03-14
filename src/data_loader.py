import pandas as pd
import numpy as np
import os
from pathlib import Path

np.random.seed(42)

def generate_mock_data():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 500 random coordinates around the world (focusing on land-like ranges)
    lats = np.random.uniform(8, 38, 500).round(2)  # India approx lats
    lons = np.random.uniform(68, 98, 500).round(2) # India approx lons
    
    dates = pd.date_range(start='2018-01-01', end='2022-12-01', freq='MS')
    
    df_base = pd.DataFrame([(lat, lon, d) for lat, lon in zip(lats, lons) for d in dates],
                           columns=['lat', 'lon', 'date'])
    
    # Geo Data
    geo_data = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'elevation': np.random.uniform(0, 3000, 500),
        'land_cover': np.random.choice([0, 1, 2], 500)
    })
    geo_data.to_csv(raw_dir / "geo_features.csv", index=False)
    
    # ERA5 Data
    era5 = df_base.copy()
    era5['temperature_2m'] = 273.15 + 25 + np.sin(era5['date'].dt.month * (2*np.pi/12)) * 10 + np.random.normal(0, 2, len(era5))
    era5['u_component_of_wind_10m'] = np.random.normal(0, 3, len(era5))
    era5['v_component_of_wind_10m'] = np.random.normal(0, 3, len(era5))
    era5['surface_pressure'] = 101325 - geo_data.set_index(['lat', 'lon']).loc[era5.set_index(['lat', 'lon']).index]['elevation'].values * 10 + np.random.normal(0, 100, len(era5))
    era5['specific_humidity'] = np.random.uniform(0.001, 0.02, len(era5))
    
    # Inject nulls
    era5.loc[np.random.choice(era5.index, 500), 'temperature_2m'] = np.nan
    era5.loc[np.random.choice(era5.index, 500), 'specific_humidity'] = np.nan
    era5.to_csv(raw_dir / "era5_meteo.csv", index=False)
    
    # MODIS AOD
    modis = df_base.copy()
    base_aod = np.random.uniform(0.1, 1.5, len(modis))
    modis['Optical_Depth_047'] = base_aod + np.random.normal(0, 0.1, len(modis))
    modis['Optical_Depth_055'] = base_aod * 0.9 + np.random.normal(0, 0.1, len(modis))
    modis['cloud_fraction'] = np.random.uniform(0, 1, len(modis))
    
    # Inject nulls
    modis.loc[np.random.choice(modis.index, 800), 'Optical_Depth_047'] = np.nan
    modis.to_csv(raw_dir / "modis_aod.csv", index=False)
    
    # PM2.5 (The target, build realistic correlations)
    pm25 = df_base.copy()
    # pm25 ~ aod * 40 - wind_speed * 2 - temperature * 0.5 + elevation_effect + noise
    merged_for_pm25 = era5.merge(modis, on=['lat', 'lon', 'date'])
    wind_speed = np.sqrt(merged_for_pm25['u_component_of_wind_10m']**2 + merged_for_pm25['v_component_of_wind_10m']**2)
    
    pm25_vals = merged_for_pm25['Optical_Depth_047'].fillna(0.5) * 50 \
                - wind_speed * 2 \
                + (merged_for_pm25['temperature_2m'] - 273.15) * 0.5 \
                + np.random.normal(10, 5, len(pm25))
                
    pm25['pm25'] = np.clip(pm25_vals, 5, 400)
    
    # Inject outliers > 500
    pm25.loc[np.random.choice(pm25.index, 100), 'pm25'] = np.random.uniform(501, 800, 100)
    
    # Inject nulls
    pm25.loc[np.random.choice(pm25.index, 200), 'pm25'] = np.nan
    
    # Save as string format to test standardizing date format
    pm25['date'] = pm25['date'].dt.strftime('%m/%d/%Y') 
    pm25.to_csv(raw_dir / "pm25_raw.csv", index=False)
    
    print("Mock data generated successfully in data/raw/")

if __name__ == "__main__":
    generate_mock_data()
