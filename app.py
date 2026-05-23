import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Changi Weather AI", layout="centered")
st.markdown("""
<style>

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #f0f9ff;
    }

    h1 {
        color: #1e3a8a;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }

    div[data-baseweb="input"], div[data-baseweb="slider"] {
        background-color: #e0f2fe !important;
        border-radius: 10px !important;
    }

    .stButton>button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        border-radius: 12px;
        border: none;
        height: 3.5em;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #0369a1;
        color: white;
    }

    hr {
        border: 0;
        height: 1px;
        background: #bae6fd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_weather_assets():
    model = joblib.load("weather_model.pkl")
    features = joblib.load("weather_features.pkl")
    return model, features

model, features = load_weather_assets()

st.markdown("<h1>☁️ Changi Precipitation Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7F8C8D;'>Enter Today's Data To Predict Tomorrow's Rainfall intensity.</p>", unsafe_allow_html=True)
st.write("---")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        m = st.slider("Month", 1, 12, 5)
        t_mean = st.number_input("Mean Temp (°C)", value=28.0)
        t_max = st.number_input("Max Temp (°C)", value=32.0)

    with col2:
        t_min = st.number_input("Min Temp (°C)", value=25.0)
        w_mean = st.number_input("Mean Wind (km/h)", value=10.0)
        w_max = st.number_input("Max Wind (km/h)", value=35.0)

if st.button("Generate Forecast"):

    input_dict = {
        'Month': m,
        'Mean Temperature (°C)': t_mean,
        'Maximum Temperature (°C)': t_max,
        'Minimum Temperature (°C)': t_min,
        'Mean Wind Speed (km/h)': w_mean,
        'Max Wind Speed (km/h)': w_max
    }
    
    # Create DataFrame and ensure column order matches 'features' list
    input_df = pd.DataFrame([input_dict])[features]
    
    # Predict
    val = model.predict(input_df)[0]
    
    st.markdown("---")
    
    # UI Results Display
    if val < 1:
        st.balloons()
        st.success(f"### ☀️ Forecast: Sunny / No Rain ({val:.2f} mm)")
    elif val < 10:
        st.warning(f"### 🌦️ Forecast: Light Showers ({val:.2f} mm)")
    else:
        st.error(f"### ⛈️ Forecast: Heavy Rain Expected ({val:.2f} mm)")

st.markdown("<br><hr><p style='text-align: center; font-size: 12px;'>Model: Random Forest Regressor</p>", unsafe_allow_html=True)