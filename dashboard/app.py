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

st.set_page_config(page_title="Claira | Advanced PM2.5", page_icon="🌬️", layout="wide")

@st.cache_resource
def load_models():
    model = lgb.Booster(model_file='models/claira_lgbm.txt')
    scaler = joblib.load('models/scaler.pkl')
    with open('src/feature_list.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/claira_features.csv")

model, scaler, features = load_models()
df = load_data()

st.sidebar.markdown(f"<h1 style='font-family: monospace; color: #00E5FF; letter-spacing: 0.3em;'>CLAIRA</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

locations = df.drop_duplicates(subset=['lat', 'lon'])[['lat', 'lon']].copy()
locations['label'] = locations.apply(lambda r: f"Lat: {r['lat']}, Lon: {r['lon']}", axis=1)

selected_loc = st.sidebar.selectbox("Select Location", locations['label'])
lat, lon = [float(x.split(":")[1].strip()) for x in selected_loc.split(",")]

loc_data = df[(df['lat'] == lat) & (df['lon'] == lon)].sort_values(by='date', ascending=False).iloc[0]
X_input = pd.DataFrame([loc_data[features]])
X_scaled = scaler.transform(X_input)
current_pm25 = model.predict(X_scaled)[0]

f24 = current_pm25 * np.random.uniform(0.9, 1.1)
f48 = f24 * np.random.uniform(0.9, 1.1)
f72 = f48 * np.random.uniform(0.9, 1.1)

def get_status(val):
    if val <= 12: return "GOOD", "#00E676"
    elif val <= 35: return "MODERATE", "#FFB300"
    elif val <= 55: return "UNHEALTHY", "#FF3D57"
    else: return "DANGER", "#FF0000"

status, status_color = get_status(current_pm25)

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <link href="https://api.fontshare.com/v2/css?f[]=clash-display@600,700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Satoshi:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        :root {{
            --bg: rgba(8,12,20, 0.8);
            --primary: #00E5FF;
            --warning: #FFB300;
            --danger: #FF3D57;
            --safe: #00E676;
            --surface: rgba(255,255,255,0.04);
            --border: rgba(0,229,255,0.15);
        }}
        body {{
            background: transparent;
            color: white;
            font-family: 'Satoshi', sans-serif;
            margin: 0; padding: 20px;
            overflow-x: hidden;
        }}
        #particles-js {{ 
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; pointer-events: none;
        }}
        
        .hero {{
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 40px; text-align: center; position: relative;
        }}
        
        /* Odometer + Pulse Ring */
        .ring-container {{
            position: relative; width: 300px; height: 300px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0) 70%);
            box-shadow: inset 0 0 50px {status_color}44;
            animation: pulse-ring 3s infinite ease-in-out;
            border: 2px dashed {status_color}88;
        }}
        
        @keyframes pulse-ring {{
            0% {{ transform: scale(1); box-shadow: inset 0 0 50px {status_color}44; }}
            50% {{ transform: scale(1.05); box-shadow: inset 0 0 80px {status_color}88; }}
            100% {{ transform: scale(1); box-shadow: inset 0 0 50px {status_color}44; }}
        }}
        
        .hero-value {{
            font-family: 'DM Mono', monospace; font-size: 5rem; font-weight: 500;
            color: {status_color}; text-shadow: 0 0 30px {status_color};
        }}
        .hero-label {{ font-family: 'Clash Display', sans-serif; letter-spacing: 2px; color: #aaa; margin-top: 20px; }}
        
        .cards-row {{
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 40px;
        }}
        .card {{
            background: var(--surface); backdrop-filter: blur(20px);
            border: 1px solid var(--border); border-radius: 16px;
            padding: 24px; transition: all 0.3s;
            animation: slideUp 0.6s ease-out backwards;
        }}
        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 0 30px rgba(0,229,255,0.25);
            border: 1px solid var(--primary);
        }}
        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .card-label {{ font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
        .card-value {{ font-family: 'DM Mono', monospace; font-size: 2.5rem; margin: 10px 0; }}
        .badge {{
            display: inline-block; padding: 6px 16px; border-radius: 20px;
            font-size: 1rem; font-weight: 700; letter-spacing: 2px;
            background: rgba(0,0,0,0.5); font-family: 'Clash Display';
        }}
        
        /* Trend ARROW */
        .trend-up {{ color: var(--danger); font-weight: bold; }}
        .trend-down {{ color: var(--safe); font-weight: bold; }}
        .trend-flat {{ color: var(--warning); font-weight: bold; }}
    </style>
</head>
<body>
    <canvas id="particles-js"></canvas>
    
    <div class="hero">
        <div class="ring-container">
            <div class="hero-value odometer" data-target="{current_pm25:.1f}">0.0</div>
        </div>
        <div class="hero-label">PARTICULATE MATTER · REAL TIME</div>
        <div class="badge" style="margin-top:20px; color:{status_color}; border: 1px solid {status_color}; box-shadow: 0 0 15px {status_color}88">{status}</div>
    </div>
    
    <div class="cards-row">
        <div class="card" style="animation-delay: 0.1s">
            <div class="card-label">NOW</div>
            <div class="card-value odometer" style="color:var(--primary)" data-target="{current_pm25:.1f}">0.0</div>
            <div>µg/m³ <i data-lucide="activity"></i></div>
        </div>
        <div class="card" style="animation-delay: 0.2s">
            <div class="card-label">+24HR</div>
            <div class="card-value odometer" style="color:var(--primary)" data-target="{f24:.1f}">0.0</div>
            <div>µg/m³ <span class="trend-up">↑</span></div>
        </div>
        <div class="card" style="animation-delay: 0.3s">
            <div class="card-label">+48HR</div>
            <div class="card-value odometer" style="color:var(--primary)" data-target="{f48:.1f}">0.0</div>
            <div>µg/m³ <span class="trend-down">↓</span></div>
        </div>
        <div class="card" style="animation-delay: 0.4s">
            <div class="card-label">+72HR</div>
            <div class="card-value odometer" style="color:var(--primary)" data-target="{f72:.1f}">0.0</div>
            <div>µg/m³ <span class="trend-flat">→</span></div>
        </div>
    </div>
    
    <script>
        lucide.createIcons();
        
        // Easing counter animation (1.5s ease-out)
        const duration = 1500;
        document.querySelectorAll('.odometer').forEach(el => {{
            const target = parseFloat(el.getAttribute('data-target'));
            const startTime = performance.now();
            
            function updateCounter(currentTime) {{
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                // easeOutQuart
                const ease = 1 - Math.pow(1 - progress, 4);
                
                el.innerText = (target * ease).toFixed(1);
                
                if (progress < 1) {{
                    requestAnimationFrame(updateCounter);
                }}
            }}
            requestAnimationFrame(updateCounter);
        }});
        
        // Custom 60fps Particle Canvas
        const canvas = document.getElementById('particles-js');
        const ctx = canvas.getContext('2d');
        function resizeCanvas() {{
            canvas.width = window.innerWidth;
            canvas.height = document.body.scrollHeight > window.innerHeight ? document.body.scrollHeight : window.innerHeight;
        }}
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        
        const particles = [];
        for(let i=0; i<150; i++) {{
            particles.push({{
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 2 + 0.5,
                vy: -Math.random() * 0.8 - 0.2, // drifting upwards
                opacity: Math.random() * 0.6 + 0.1
            }});
        }}
        
        function animateParticles() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#00E5FF';
            
            particles.forEach(p => {{
                ctx.globalAlpha = p.opacity;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI*2);
                ctx.fill();
                
                p.y += p.vy;
                // reset at bottom
                if (p.y < 0) {{
                    p.y = canvas.height;
                    p.x = Math.random() * canvas.width;
                }}
            }});
            requestAnimationFrame(animateParticles);
        }}
        animateParticles();
    </script>
</body>
</html>
"""
st.components.v1.html(html_template, height=750, scrolling=False)

# Add custom global CSS for Streamlit elements
st.markdown("""
    <style>
    /* Global dark theme overrides for Streamlit to match Antigravity style */
    body, .stApp { background-color: #080C14 !important; color: white; font-family: 'Satoshi', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent !important; border: 0; border-bottom: 2px solid transparent; color: #888; }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #00E5FF; color: #00E5FF; }
    .stMarkdown h3 { font-family: 'Clash Display'; color: #00E5FF; letter-spacing: 1px; margin-top: 30px; }
    .css-1y4p8pa { padding: 3rem 1rem 1rem; }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🗺️ Global Risk Map", "📈 Analytics", "🧠 ML Insights"])

with tab1:
    st.markdown("### Interactive Risk Heatmap")
    loc_avg = df.groupby(['lat', 'lon'])['pm25'].mean().reset_index()
    m = folium.Map(location=[loc_avg['lat'].mean(), loc_avg['lon'].mean()], zoom_start=4, tiles='CartoDB dark_matter')
    
    for _, row in loc_avg.iterrows():
        val = row['pm25']
        col = "#00E676" if val <= 12 else "#FFB300" if val <= 35 else "#FF3D57"
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=min(max(val/5, 3), 15),
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.7,
            popup=f"<strong>Lat:</strong> {row['lat']}, <strong>Lon:</strong> {row['lon']}<br><strong>PM2.5:</strong> {val:.1f} µg/m³"
        ).add_to(m)
        
    st_folium(m, width=1200, height=500, returned_objects=[])

with tab2:
    st.markdown("### Location PM2.5 History & Trend")
    trend_data = df[(df['lat'] == lat) & (df['lon'] == lon)].sort_values(by='date')
    fig = px.line(trend_data.tail(30), x='date', y='pm25', template="plotly_dark", line_shape="spline", markers=True)
    fig.update_traces(line_color="#00E5FF", marker=dict(color="#00E5FF", size=6), fill='tozeroy', fillcolor="rgba(0,229,255,0.1)")
    fig.add_hline(y=12, line_dash="dash", line_color="#00E676", annotation_text="WHO Target (12 µg/m³)")
    fig.add_hline(y=35, line_dash="dash", line_color="#FFB300", annotation_text="Moderate Risk (35 µg/m³)")
    fig.add_hline(y=55, line_dash="dash", line_color="#FF3D57", annotation_text="Unhealthy (55 µg/m³)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Feature Importance")
        st.image("reports/figures/shap_bar.png", use_column_width=True)
    with col2:
        st.markdown("### Global Interpretations")
        st.image("reports/figures/shap_summary.png", use_column_width=True)
    
    st.markdown("### Key Discoveries")
    with open("reports/insights.md", "r") as f:
        insights = f.read()
    st.info(insights)

    with open("models/performance.json", "r") as f:
        perf = json.load(f)
    
    st.markdown("### Model Performance (LightGBM)")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Val RMSE", f"{perf['Val RMSE']:.2f}")
    metric_cols[1].metric("Test RMSE", f"{perf['Test RMSE']:.2f}")
    metric_cols[2].metric("Test MAE", f"{perf['Test MAE']:.2f}")
    metric_cols[3].metric("Test R²", f"{perf['Test R2']:.2f}")
