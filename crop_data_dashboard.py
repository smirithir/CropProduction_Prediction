
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Load model & data ---
model = joblib.load("production_model.pkl")
df = pd.read_csv("cleaned_data.csv")

# --- Streamlit Configuration ---
st.set_page_config(page_title="🌾 Crop Production Predictor", layout="wide")

# --- Sidebar: Input Panel ---
st.sidebar.header("🔍 Enter Crop Info for Prediction")
area = st.sidebar.selectbox("🌍 Region", df['Area'].unique())
item = st.sidebar.selectbox("🌿 Crop", df['Item'].unique())
year = st.sidebar.slider("📅 Year", int(df['Year'].min()), int(df['Year'].max()), step=1)
area_harvested = st.sidebar.number_input("🧱 Area Harvested (ha)", min_value=0.0)
yield_val = st.sidebar.number_input("🌱 Yield (kg/ha)", min_value=0.0)

if st.sidebar.button("🔮 Predict Production"):
    # Encode inputs
    area_code = df[df['Area'] == area]['Area_Code'].iloc[0]
    item_code = df[df['Item'] == item]['Item_Code'].iloc[0]
    X_input = [[area_code, item_code, year, area_harvested, yield_val]]
    prediction = model.predict(X_input)[0]
    st.sidebar.success(f"Predicted Production: {prediction:.2f} tons")

# --- Dashboard Layout ---
st.title("📊 Crop Production Dashboard")
st.markdown("Analyze trends and predict agricultural output based on region, crop type, and land use.")

# --- 1. Production Trend Line ---
st.subheader("📈 Yearly Production Trend by Crop")
trend_df = df.groupby(['Year', 'Item'])['Production'].mean().reset_index()
fig_trend = px.line(trend_df[trend_df['Item'] == item], x='Year', y='Production', title=f"Yearly Production Trend for {item}")
st.plotly_chart(fig_trend, use_container_width=True)

# --- 2. Top Productive Crops ---
st.subheader("🌾 Top Productive Crops (Production / Area)")
top_df = df.groupby('Item')['Productivity'].mean().sort_values(ascending=False).reset_index().head(10)
fig_bar = px.bar(top_df, x='Item', y='Productivity', title="Top 10 Productive Crops", color='Productivity')
st.plotly_chart(fig_bar, use_container_width=True)

# --- 4. Correlation Heatmap Image ---
st.subheader("🔗 Input-Output Correlation")
st.image("correlation_heatmap.png", use_column_width=True)

# --- 5. Productivity Comparison ---
st.subheader("📊 Productivity Comparison")
st.image("productivity_comparison.png", use_column_width=True)
