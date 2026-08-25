
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Crop Production Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# STYLING
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; }

    .main { background-color: #F7FAF7; }

    .hero-banner {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 45%, #66BB6A 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(27,94,32,0.25);
    }
    .hero-banner h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .hero-banner p { margin: 0.4rem 0 0 0; font-size: 1.02rem; opacity: 0.92; }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 5px solid #2E7D32;
        text-align: left;
    }
    .metric-card .label { font-size: 0.82rem; color: #5f6b5f; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
    .metric-card .value { font-size: 1.6rem; color: #1B5E20; font-weight: 700; margin-top: 0.15rem; }

    .prediction-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #66BB6A;
        margin-top: 1rem;
    }
    .prediction-box .pred-label { font-size: 1rem; color: #2E7D32; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .prediction-box .pred-value { font-size: 3rem; color: #1B5E20; font-weight: 700; margin: 0.3rem 0; }
    .prediction-box .pred-sub { color: #4d5d4d; font-size: 0.95rem; }

    .section-tag {
        display: inline-block;
        background: #E8F5E9;
        color: #1B5E20;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 0.6rem;
        letter-spacing: 0.4px;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 1.1rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# DATA & MODEL LOADING (cached so it only runs once per session)

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_crop_data.csv")
    return df

@st.cache_resource
def load_model_artifacts():
    model = joblib.load("production_model.pkl")
    le_area = joblib.load("le_area.pkl")
    le_item = joblib.load("le_item.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, le_area, le_item, feature_cols

try:
    df = load_data()
    model, le_area, le_item, feature_cols = load_model_artifacts()
    DATA_OK = True
except FileNotFoundError as e:
    DATA_OK = False
    MISSING_FILE = str(e)


st.markdown("""
# ----------------------------------------------------------------------------
# HERO BANNER
# ----------------------------------------------------------------------------
<div class="hero-banner">
    <h1>🌾 Crop Production Predictor</h1>
    <p>Forecast agricultural production worldwide and explore global crop trends — built on FAOSTAT data (2019–2023).</p>
</div>
""", unsafe_allow_html=True)

if not DATA_OK:
    st.error(
        f"Required file not found: {MISSING_FILE}\n\n"
        "Make sure `cleaned_crop_data.csv`, `production_model.pkl`, `le_area.pkl`, "
        "`le_item.pkl`, and `feature_cols.pkl` (all produced by the notebook) are in the same folder as `app.py`."
    )
    st.stop()

# TOP-LEVEL 
num_crops = df['Item'].nunique()
num_countries = df['Area'].nunique()
total_production = df['Production_tons'].sum()
year_min, year_max = int(df['Year'].min()), int(df['Year'].max())

k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (k1, "Crop Types Tracked", f"{num_crops}"),
    (k2, "Countries & Regions", f"{num_countries}"),
    (k3, "Total Production Recorded", f"{total_production/1e9:.1f} B tons"),
    (k4, "Years Covered", f"{year_min}–{year_max}"),
]
for col, label, value in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# TABS
tab_predict, tab_world, tab_trends, tab_compare = st.tabs(
    [" Predict Production"]) 
# " World Crop Overview", " Trends Over Time", " Compare Crops & Regions"]
# )

# TAB 1: PREDICTION

with tab_predict:
    st.markdown('<span class="section-tag">PREDICTION TOOL</span>', unsafe_allow_html=True)
    st.subheader("Get an instant production forecast")
    st.write("Choose a country/region and crop, then enter the growing conditions to predict total production.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.form("prediction_form"):
            st.markdown("##### 📍 Location & Crop")
            c1, c2 = st.columns(2)
            with c1:
                area_choice = st.selectbox("Country / Region", options=sorted(df['Area'].unique()), index=sorted(df['Area'].unique()).index("India") if "India" in df['Area'].unique() else 0)
            with c2:
                item_choice = st.selectbox("Crop / Item", options=sorted(df['Item'].unique()), index=0)

            year_choice = st.slider("Year", min_value=2015, max_value=2030, value=2024, help="You can forecast slightly beyond the historical data range (2019–2023).")

            st.markdown("##### 🌱 Growing Conditions")
            c3, c4 = st.columns(2)
            with c3:
                default_area = float(df[df['Item'] == item_choice]['Area_Harvested_ha'].median())
                area_harvested = st.number_input(
                    "Area Harvested (hectares)",
                    min_value=0.1,
                    value=round(default_area, 1),
                    step=100.0,
                    help="Total land area used to grow this crop."
                )
            with c4:
                default_yield = float(df[df['Item'] == item_choice]['Yield_kg_ha'].median())
                yield_val = st.number_input(
                    "Yield (kg per hectare)",
                    min_value=0.1,
                    value=round(default_yield, 1),
                    step=10.0,
                    help="Expected output per hectare of land."
                )

            submitted = st.form_submit_button("🔮 Predict Production", use_container_width=True)

    with right:
        if submitted:
            try:
                area_enc = le_area.transform([area_choice])[0]
            except ValueError:
                area_enc = 0
            try:
                item_enc = le_item.transform([item_choice])[0]
            except ValueError:
                item_enc = 0

            interaction = area_harvested * yield_val
            X_input = pd.DataFrame([[area_enc, item_enc, year_choice, area_harvested, yield_val, interaction]],
                                    columns=feature_cols)
            prediction = model.predict(X_input)[0]
            prediction = max(prediction, 0)

            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">Predicted Production</div>
                <div class="pred-value">{prediction:,.0f} tons</div>
                <div class="pred-sub">{item_choice} · {area_choice} · {year_choice}</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            hist = df[(df['Area'] == area_choice) & (df['Item'] == item_choice)].sort_values('Year')
            if not hist.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['Year'], y=hist['Production_tons'], mode='lines+markers',
                                          name='Historical', line=dict(color='#2E7D32', width=3)))
                fig.add_trace(go.Scatter(x=[year_choice], y=[prediction], mode='markers',
                                          name='Prediction', marker=dict(color='#E65100', size=16, symbol='star')))
                fig.update_layout(title=f"Historical vs Predicted Production — {item_choice} in {area_choice}",
                                   xaxis_title="Year", yaxis_title="Production (tons)",
                                   plot_bgcolor='white', height=380, margin=dict(t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical record for this exact crop/region combination — showing prediction only.")
        else:
            st.markdown("""
            <div style="background:white; border-radius:14px; padding:2.5rem; text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-top: 1.9rem;">
                <span style="font-size:3rem;">🌾</span>
                <p style="color:#5f6b5f; margin-top:0.8rem;">Fill in the form and click <b>Predict Production</b> to see your forecast here.</p>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown("---")
# st.caption("Data source: FAOSTAT — Crops and Livestock Products (2019–2023) · Built with Streamlit, scikit-learn & Plotly")
