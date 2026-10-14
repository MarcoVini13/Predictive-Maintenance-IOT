import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Configuración de página
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Industrial IoT - Predictive Maintenance Dashboard")
st.markdown("Real-time telemetry analysis & predictive failure detection for manufacturing assets.")

# Cargar modelo y datos
@st.cache_resource
def load_assets():
    model = joblib.load('models/maintenance_model.pkl')
    df = pd.read_csv('data/sensor_data.csv')
    return model, df

try:
    model, df = load_assets()
except Exception as e:
    st.error("Por favor ejecuta primero 'src/generate_data.py' y 'src/train_model.py'")
    st.stop()

# Layout con columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎛️ Live Sensor Simulation")
    
    temp = st.slider("Temperature (°C)", 30.0, 120.0, 72.0)
    vib = st.slider("Vibration (mm/s)", 0.5, 10.0, 3.2)
    press = st.slider("Pressure (PSI)", 50.0, 130.0, 95.0)
    hours = st.slider("Operating Hours", 100, 10000, 2500)
    
    # Predicción
    input_data = pd.DataFrame([[temp, vib, press, hours]], 
                             columns=['temperature', 'vibration', 'pressure', 'operating_hours'])
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1] * 100
    
    st.divider()
    st.subheader("🚨 Risk Status")
    if prediction == 1:
        st.error(f"**CRITICAL WARNING:** High Failure Risk ({proba:.1f}%)")
        st.caption("Action Required: Schedule Immediate Maintenance Check.")
    else:
        st.success(f"**OPERATIONAL:** Machine Healthy (Failure Risk: {proba:.1f}%)")
        st.caption("Status: All parameters within safe margins.")

with col2:
    st.subheader("📈 Historical Telemetry & Failure Patterns")
    
    fig = px.scatter(
        df, x='temperature', y='vibration', color=df['failure'].map({0: 'Healthy', 1: 'Failed'}),
        title="Vibration vs Temperature Correlation",
        labels={'temperature': 'Temperature (°C)', 'vibration': 'Vibration (mm/s)', 'color': 'Status'},
        color_discrete_map={'Healthy': '#2ecc71', 'Failed': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Métricas rápidas
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Analyzed Machines", len(df))
m2.metric("Failure Rate", f"{(df['failure'].mean() * 100):.1f}%")
m3.metric("Avg Temperature", f"{df['temperature'].mean():.1f} °C")
m4.metric("Avg Vibration", f"{df['vibration'].mean():.2f} mm/s")
