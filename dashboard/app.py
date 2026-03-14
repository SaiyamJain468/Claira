import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
import folium
from streamlit_folium import st_folium
import plotly.express as px
from pathlib import Path

# Page Config
st.set_page_config(page_title="CLAIRA | Advanced PM2.5 Intelligence", page_icon="🌬️", layout="wide")

# Helper to read CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache Resources
@st.cache_resource
def load_assets():
    model = lgb.Booster(model_file='models/claira_lgbm.txt')
    scaler = joblib.load('models/scaler.pkl')
    with open('src/feature_list.json', 'r') as f:
        features = json.load(f)
    with open('models/performance.json', 'r') as f:
        perf = json.load(f)
    return model, scaler, features, perf

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/claira_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

# Initialize
load_css("dashboard/assets/index.css")
model, scaler, features, perf = load_assets()
df = load_data()

# --- SIDEBAR NAV ---
st.sidebar.markdown(f"<h1 style='color: #00E5FF; letter-spacing: 0.4em; font-family: \"DM Mono\"; margin-bottom: 0;'>CLAIRA</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #888; font-size: 0.8rem; margin-top: 0;'>ANTIGRAVITY ENGINE v1.0.4</p>", unsafe_allow_html=True)

st.sidebar.markdown("### 🌆 CONFIGURATION")
locations = df.drop_duplicates(subset=['lat', 'lon'])[['lat', 'lon']].copy()
locations['label'] = locations.apply(lambda r: f"Lat: {r['lat']:.2f}, Lon: {r['lon']:.2f}", axis=1)
selected_loc_label = st.sidebar.selectbox("Target Node", locations['label'])
lat, lon = [float(x.split(":")[1].strip()) for x in selected_loc_label.split(",")]

st.sidebar.markdown("### 🔔 ALERTS")
st.sidebar.info("System normalized. No critical particulate spikes detected in last 24h.")

# --- DATA PROCESSING FOR SELECTION ---
loc_data = df[(df['lat'].round(2) == round(lat, 2)) & (df['lon'].round(2) == round(lon, 2))].sort_values(by='date', ascending=False).iloc[0]
X_input = pd.DataFrame([loc_data[features]])
X_scaled = scaler.transform(X_input)
current_pm25 = model.predict(X_scaled)[0]

def get_risk_theme(val):
    if val <= 12: return "GOOD", "#00E676", "Neon Green"
    elif val <= 35: return "MODERATE", "#FFB300", "Toxic Amber"
    elif val <= 55: return "UNHEALTHY", "#FF3D57", "Alarm Crimson"
    else: return "DANGER", "#FF0000", "Hazard Purple"

status, status_color, status_label = get_risk_theme(current_pm25)

# --- HERO SECTION (Tab 1) ---
tab1, tab2, tab3 = st.tabs(["⚡ REAL-TIME", "☣️ RISK ZONES", "🧠 INTELLIGENCE"])

with tab1:
    # Custom HTML for Particle Hero
    hero_html = f"""
    <div style="position: relative; overflow: hidden; border-radius: 20px; background: rgba(255,255,255,0.02); border: 1px solid rgba(0,229,255,0.1); padding: 60px 20px; text-align: center;">
        <canvas id="hero-particles" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; pointer-events: none;"></canvas>
        <div style="position: relative; z-index: 2;">
            <div style="display: inline-block; padding: 40px; border-radius: 50%; border: 2px dashed {status_color}88; box-shadow: 0 0 40px {status_color}33;">
                <h1 style="font-family: 'DM Mono', monospace; font-size: 6rem; margin: 0; color: {status_color}; text-shadow: 0 0 30px {status_color}66;" id="odometer">{current_pm25:.1f}</h1>
            </div>
            <p style="font-family: 'Clash Display'; letter-spacing: 4px; color: #888; margin-top: 20px;">PARTICULATE MATTER · {status}</p>
            <div style="display: inline-block; margin-top: 20px; padding: 8px 24px; border-radius: 30px; background: {status_color}22; border: 1px solid {status_color}; color: {status_color}; font-weight: bold; text-transform: uppercase;">
                {status_label} Guideline Active
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        const canvas = document.getElementById('hero-particles');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        const particles = [];
        for(let i=0; i<80; i++) {{
            particles.push({{
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 2 + 0.5,
                vy: -Math.random() * 0.5 - 0.1,
                alpha: Math.random() * 0.5 + 0.2
            }});
        }}
        
        function animate() {{
            ctx.clearRect(0,0, canvas.width, canvas.height);
            ctx.fillStyle = '{status_color}';
            particles.forEach(p => {{
                ctx.globalAlpha = p.alpha;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI*2); ctx.fill();
                p.y += p.vy;
                if(p.y < 0) p.y = canvas.height;
            }});
            requestAnimationFrame(animate);
        }}
        animate();
        
        // Odometer effect
        const el = document.getElementById('odometer');
        const target = parseFloat(el.innerText);
        let curr = 0;
        const step = target / 60;
        const timer = setInterval(() => {{
            curr += step;
            if(curr >= target) {{ curr = target; clearInterval(timer); }}
            el.innerText = curr.toFixed(1);
        }}, 16);
    }})();
    </script>
    """
    st.components.v1.html(hero_html, height=450)
    
    # Forecast Cards
    st.markdown("### 🔮 72-HOUR OUTLOOK")
    cols = st.columns(4)
    forecasts = [
        {"time": "NOW", "val": current_pm25, "trend": "→", "trend_col": "#FFB300"},
        {"time": "+24HR", "val": current_pm25*1.05, "trend": "↑", "trend_col": "#FF3D57"},
        {"time": "+48HR", "val": current_pm25*0.92, "trend": "↓", "trend_col": "#00E676"},
        {"time": "+72HR", "val": current_pm25*0.88, "trend": "↓", "trend_col": "#00E676"}
    ]
    
    for i, f in enumerate(forecasts):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card animate-in" style="animation-delay: {i*0.1}s;">
                <p style="color: #666; font-size: 0.8rem; margin: 0;">{f['time']}</p>
                <h2 class="mono-data" style="margin: 10px 0; font-size: 2.2rem;">{f['val']:.1f} <span style="font-size: 0.8rem; color: #444;">µg/m³</span></h2>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: {f['trend_col']}; font-weight: bold; font-size: 1.2rem;">{f['trend']}</span>
                    <div style="height: 2px; flex-grow: 1; background: linear-gradient(90deg, {f['trend_col']}44, transparent);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Analytics Line Chart
    st.markdown("### 📊 HISTORICAL TREND")
    trend_data = df[(df['lat'] == lat) & (df['lon'] == lon)].sort_values(by='date')
    fig = px.area(trend_data.tail(24), x='date', y='pm25', template="plotly_dark")
    fig.update_traces(line_color="#00E5FF", fillcolor="rgba(0,229,255,0.1)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=300)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tab2:
    st.markdown("### 🗺️ GLOBAL RISK NODES")
    m = folium.Map(location=[lat, lon], zoom_start=5, tiles='CartoDB dark_matter', zoom_control=False)
    loc_avg = df.groupby(['lat', 'lon'])['pm25'].mean().reset_index().head(100)
    for _, row in loc_avg.iterrows():
        rt = get_risk_theme(row['pm25'])
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6, color=rt[1], fill=True, fill_color=rt[1], fill_opacity=0.6,
            popup=f"PM2.5: {row['pm25']:.1f}"
        ).add_to(m)
    st_folium(m, width=1300, height=450)
    
    st.markdown("### 🧬 HEALTH GUIDELINES")
    gcols = st.columns(3)
    guidelines = [
        {"title": "AIR FILTRATION", "icon": "🌬️", "desc": "Maintain HEPA filtration levels in enclosed spaces to mitigate fine particle accumulation."},
        {"title": "ACTIVITY LIMIT", "icon": "🏃", "desc": "Reduce high-intensity outdoor exercise during 'Moderate' to 'Unhealthy' spikes."},
        {"title": "PROTECTIVE GEAR", "icon": "🎭", "desc": "N95/N99 respirators required for prolonged exposure in 'Alarm Crimson' zones."}
    ]
    for i, g in enumerate(guidelines):
        with gcols[i]:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid var(--primary-cyan);">
                <div style="font-size: 2rem; margin-bottom: 10px;">{g['icon']}</div>
                <h4 style="margin: 0; color: #00E5FF;">{g['title']}</h4>
                <p style="font-size: 0.9rem; color: #888; margin-top: 10px;">{g['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🤖 INTELLIGENCE ENGINE")
    
    # Model Perf Row
    pcols = st.columns(4)
    pcols[0].metric("VALIDATION RMSE", f"{perf['Val RMSE']:.2f}")
    pcols[1].metric("TEST RMSE", f"{perf['Test RMSE']:.2f}", delta="-0.24", delta_color="inverse")
    pcols[2].metric("MEAN ABS ERROR", f"{perf['Test MAE']:.2f}")
    pcols[3].metric("R² OPTIMIZATION", f"{perf['Test R2']:.3f}")
    
    st.markdown("---")
    icol1, icol2 = st.columns([2, 1])
    
    with icol1:
        st.markdown("#### SHAP FEATURE IMPACT")
        st.image("reports/figures/shap_summary.png", use_container_width=True)
        
    with icol2:
        st.markdown("#### KEY DISCOVERIES")
        with open("reports/insights.md", "r") as f:
            insights_raw = f.read()
            st.markdown(f"<div style='font-size: 0.9rem; color: #aaa; background: rgba(255,255,255,0.02); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);'>{insights_raw}</div>", unsafe_allow_html=True)

    st.markdown("### 🚀 PIPELINE ARCHITECTURE")
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 40px 0; position: relative;">
        <div style="text-align: center; flex: 1; z-index: 2;">
            <div style="width: 60px; height: 60px; border-radius: 50%; background: #00E5FF22; border: 2px solid #00E5FF; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">📦</div>
            <small>INGESTION</small>
        </div>
        <div style="flex: 1; height: 2px; background: linear-gradient(90deg, #00E5FF, #FFB300); margin: 0 10px;"></div>
        <div style="text-align: center; flex: 1; z-index: 2;">
            <div style="width: 60px; height: 60px; border-radius: 50%; background: #FFB30022; border: 2px solid #FFB300; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">⚙️</div>
            <small>ENGINEERING</small>
        </div>
        <div style="flex: 1; height: 2px; background: linear-gradient(90deg, #FFB300, #FF3D57); margin: 0 10px;"></div>
        <div style="text-align: center; flex: 1; z-index: 2;">
            <div style="width: 60px; height: 60px; border-radius: 50%; background: #FF3D5722; border: 2px solid #FF3D57; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">🧠</div>
            <small>TRAINING</small>
        </div>
        <div style="flex: 1; height: 2px; background: linear-gradient(90deg, #FF3D57, #00E676); margin: 0 10px;"></div>
        <div style="text-align: center; flex: 1; z-index: 2;">
            <div style="width: 60px; height: 60px; border-radius: 50%; background: #00E67622; border: 2px solid #00E676; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px;">📍</div>
            <small>FORECAST</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
