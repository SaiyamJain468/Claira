import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from pathlib import Path

def run_eda() -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the engineered features.
    Generates distribution plots, trend analysis, and a Folium heatmap.
    
    Returns:
        None
    """
    proc_dir = Path("data/processed")
    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(proc_dir / "claira_features.csv")
    
    # T074: Histogram
    plt.figure()
    sns.histplot(df['pm25'], log_scale=True, bins=50)
    plt.title("PM2.5 Distribution (Log Scale)")
    plt.savefig(reports_dir / "pm25_dist.png")
    plt.close()
    
    # T075: Boxplot by season
    plt.figure()
    sns.boxplot(x='season', y='pm25', data=df)
    plt.title("PM2.5 by Season")
    plt.savefig(reports_dir / "pm25_season.png")
    plt.close()
    
    # T076: Boxplot by land cover (is_urban)
    plt.figure()
    sns.boxplot(x='is_urban', y='pm25', data=df)
    plt.title("PM2.5 by Urban/Rural")
    plt.savefig(reports_dir / "pm25_urban_rural.png")
    plt.close()
    
    # T077: Monthly PM2.5 trend
    plt.figure()
    df.groupby('date')['pm25'].mean().plot()
    plt.title("Global Monthly PM2.5 Trend")
    plt.savefig(reports_dir / "pm25_trend.png")
    plt.close()
    
    # T078: Correlation of all features (re-run with engineered features)
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr[['pm25']].sort_values(by='pm25', ascending=False).head(15), annot=True, cmap='coolwarm')
    plt.title("Top Correlations with PM2.5")
    plt.savefig(reports_dir / "correlation.png")
    plt.close()
    
    # T079: Scatter AOD vs PM2.5 by season
    plt.figure()
    sns.scatterplot(x='aod_mean', y='pm25', hue='season', data=df.sample(2000), alpha=0.5)
    plt.title("AOD vs PM2.5 by Season")
    plt.savefig(reports_dir / "aod_vs_pm25.png")
    plt.close()
    
    # T080: Scatter Wind Speed vs PM2.5
    plt.figure()
    sns.scatterplot(x='wind_speed', y='pm25', data=df.sample(2000), alpha=0.5)
    plt.title("Wind Speed vs PM2.5")
    plt.savefig(reports_dir / "wind_vs_pm25.png")
    plt.close()
    
    # T081: Folium Heatmap
    loc_avg = df.groupby(['lat', 'lon'])['pm25'].mean().reset_index()
    m = folium.Map(location=[loc_avg['lat'].mean(), loc_avg['lon'].mean()], zoom_start=4)
    HeatMap(data=loc_avg[['lat', 'lon', 'pm25']].values.tolist(), radius=15).add_to(m)
    m.save(reports_dir / "pm25_heatmap.html")
    
    # T082: Top 10 most polluted
    top_10 = loc_avg.sort_values(by='pm25', ascending=False).head(10)
    top_10.to_csv(reports_dir / "top_10_polluted.csv", index=False)
    print("Top 10 Polluted:\n", top_10)
    
    # T083: Top 10 cleanest
    bottom_10 = loc_avg.sort_values(by='pm25', ascending=True).head(10)
    bottom_10.to_csv(reports_dir / "top_10_cleanest.csv", index=False)
    print("Top 10 Cleanest:\n", bottom_10)
    
    print("EDA Visualizations generated in reports/figures/")

if __name__ == "__main__":
    run_eda()
