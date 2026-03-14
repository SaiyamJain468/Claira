import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def clean_data() -> None:
    """
    Loads raw datasets, handles missing values, removes outliers, 
    and merges data into a standardized format.
    
    Returns:
        None
    """
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/figures")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # T036: Load all 4 CSVs
    pm25 = pd.read_csv(raw_dir / "pm25_raw.csv")
    era5 = pd.read_csv(raw_dir / "era5_meteo.csv")
    modis = pd.read_csv(raw_dir / "modis_aod.csv")
    geo = pd.read_csv(raw_dir / "geo_features.csv")
    
    # T037: Check missing value %
    print("PM2.5 missing %:\n", pm25.isnull().mean() * 100)
    
    # T038: Drop rows where pm25 is null
    pm25 = pm25.dropna(subset=['pm25'])
    
    # T039: Cap PM2.5 outliers (remove > 500)
    pm25 = pm25[pm25['pm25'] <= 500]
    
    # T043: Standardize date format YYYY-MM-01
    pm25['date'] = pd.to_datetime(pm25['date']).dt.strftime('%Y-%m-01')
    era5['date'] = pd.to_datetime(era5['date']).dt.strftime('%Y-%m-01')
    modis['date'] = pd.to_datetime(modis['date']).dt.strftime('%Y-%m-01')
    
    # T044: Round lat/lon to 2 decimal places
    pm25['lat'] = pm25['lat'].round(2)
    pm25['lon'] = pm25['lon'].round(2)
    era5['lat'] = era5['lat'].round(2)
    era5['lon'] = era5['lon'].round(2)
    modis['lat'] = modis['lat'].round(2)
    modis['lon'] = modis['lon'].round(2)
    geo['lat'] = geo['lat'].round(2)
    geo['lon'] = geo['lon'].round(2)

    # T040: Fill era5 temp using monthly mean per region
    era5['month'] = pd.to_datetime(era5['date']).dt.month
    era5['temperature_2m'] = era5['temperature_2m'].fillna(era5.groupby('month')['temperature_2m'].transform('mean'))
    era5 = era5.drop(columns=['month'])
    
    # T041: Fill humidity using linear interpolation within lat/lon
    era5 = era5.sort_values(by=['lat', 'lon', 'date'])
    era5['specific_humidity'] = era5.groupby(['lat', 'lon'])['specific_humidity'].transform(lambda x: x.interpolate(method='linear').bfill().ffill())
    
    # T042: Fill AOD using 3-month rolling mean per location
    modis = modis.sort_values(by=['lat', 'lon', 'date'])
    modis['Optical_Depth_047'] = modis.groupby(['lat', 'lon'])['Optical_Depth_047'].transform(lambda x: x.fillna(x.rolling(3, min_periods=1).mean().bfill()))
    modis = modis.dropna(subset=['Optical_Depth_047', 'Optical_Depth_055'])
    
    # T045: Merge all 4 DFs
    merged = pm25.merge(era5, on=['lat', 'lon', 'date'], how='inner')
    merged = merged.merge(modis, on=['lat', 'lon', 'date'], how='inner')
    merged = merged.merge(geo, on=['lat', 'lon'], how='inner')
    
    # T046: shape check
    print("Shape after merge:", merged.shape)
    
    # T048: Check duplicate rows
    merged = merged.drop_duplicates(subset=['lat', 'lon', 'date'])
    
    # T049: Reset index
    merged = merged.reset_index(drop=True)
    
    # T047: Print correlation with PM2.5, save heatmap
    print(merged.corr(numeric_only=True)['pm25'].sort_values())
    plt.figure(figsize=(10, 8))
    corr = merged.corr(numeric_only=True)
    sns.heatmap(corr[['pm25']].sort_values(by='pm25', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation with PM2.5")
    plt.tight_layout()
    plt.savefig(report_dir / "correlation.png")
    
    # T050: Save processed Data
    merged.to_csv(proc_dir / "claira_clean.csv", index=False)
    print("Data cleaning completed. Saved to claira_clean.csv.")

if __name__ == "__main__":
    clean_data()
