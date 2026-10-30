from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st
import textwrap

st.set_page_config(
    page_title="Atmosphere | Changi Weather Lab",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load external CSS
css_path = Path(__file__).with_name("style.css")
st.markdown(
    f"<style>{css_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True
)

# LOAD TRAINED MODEL

@st.cache_resource
def load_assets():
    root = Path(__file__).parent

    model = joblib.load(root / "weather_model.pkl")
    features = joblib.load(root / "weather_features.pkl")

    return model, features


try:
    model, features = load_assets()
except (FileNotFoundError, OSError):
    st.error(
        "Model assets could not be loaded. "
        "Place weather_model.pkl and weather_features.pkl alongside app.py."
    )
    st.stop()


# BRAND + HERO + WEATHER OUTLOOK
st.html("""
<div class="app-top">
    <div class="brand">
        <div class="brand-symbol">≋</div>
        <div>
            <div class="brand-name">atmosphere<span>.</span></div>
            <div class="brand-kicker">CHANGI WEATHER LAB</div>
        </div>
    </div>
    <div class="place">
        <span class="place-dot"></span> Changi, Singapore
    </div>
</div>

<div class="intro">
    <div class="eyebrow">
        <span></span> WEATHER INTELLIGENCE, SIMPLIFIED
    </div>
    <h1>A clearer outlook for <em>tomorrow.</em></h1>
    <p>
        Understand tomorrow's rainfall using today's weather observations,
        powered by machine learning.
    </p>
</div>

<div class="outlook">
    <div class="outlook-copy">
        <div class="outlook-pill">YOUR NEXT-DAY OUTLOOK</div>
        <h2>Forecast smarter<span>.</span></h2>
        <p>
            From six simple weather signals to a clearer rainfall estimate.
        </p>
    </div>

    <div class="weather-art" aria-hidden="true">
        <div class="sun"></div>
        <div class="cloud cloud-back"></div>
        <div class="cloud cloud-front"></div>
        <div class="art-spark a">✦</div>
        <div class="art-spark b">✧</div>
    </div>

    <div class="mini-cards">
        <div class="mini-card">
            <span class="mini-icon">◷</span>
            <strong>Tomorrow</strong>
            <small>Forecast horizon</small>
        </div>

        <div class="mini-card">
            <span class="mini-icon">℃</span>
            <strong>Temperature</strong>
            <small>Weather inputs</small>
        </div>

        <div class="mini-card">
            <span class="mini-icon">≋</span>
            <strong>Wind speed</strong>
            <small>Weather inputs</small>
        </div>
    </div>
</div>
""")



# INPUT SECTION
st.html("""
<div class="section-lead">
    <div>
        <h3>Set weather conditions</h3>
        <p>Enter today's observations to calculate tomorrow's rainfall.</p>
    </div>
    <div class="section-mark">☷</div>
</div>
""")


months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

with st.form("weather_form", border=True):

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        month = st.selectbox(
            "Month",
            months,
            index=4
        )

        mean_temp = st.number_input(
            "Mean temperature (°C)",
            value=28.0,
            step=0.1,
            format="%.1f"
        )

        max_temp = st.number_input(
            "Maximum temperature (°C)",
            value=32.0,
            step=0.1,
            format="%.1f"
        )

    with col2:
        min_temp = st.number_input(
            "Minimum temperature (°C)",
            value=25.0,
            step=0.1,
            format="%.1f"
        )

        mean_wind = st.number_input(
            "Mean wind speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=0.1,
            format="%.1f"
        )

        max_wind = st.number_input(
            "Maximum wind speed (km/h)",
            min_value=0.0,
            value=35.0,
            step=0.1,
            format="%.1f"
        )

    submitted = st.form_submit_button(
        "✦  Generate rainfall forecast   →",
        use_container_width=True
    )


# PREDICTION
if submitted:

    if not min_temp <= mean_temp <= max_temp:
        st.error(
            "Temperature values must satisfy: "
            "Minimum ≤ Mean ≤ Maximum."
        )

    elif mean_wind > max_wind:
        st.error(
            "Mean wind speed cannot be greater "
            "than maximum wind speed."
        )

    else:
        input_data = {
            "Month": months.index(month) + 1,
            "Mean Temperature (°C)": mean_temp,
            "Maximum Temperature (°C)": max_temp,
            "Minimum Temperature (°C)": min_temp,
            "Mean Wind Speed (km/h)": mean_wind,
            "Max Wind Speed (km/h)": max_wind
        }

        try:
            # Preserve original training feature order
            frame = pd.DataFrame([input_data])[features]

            predicted = max(
                0.0,
                float(model.predict(frame)[0])
            )

            # Classify predicted rainfall
            if predicted < 1:
                label = "Minimal rainfall"
                icon = "☀"
                category = "clear"
                note = (
                    "Little or no rainfall is estimated "
                    "for tomorrow."
                )

            elif predicted < 10:
                label = "Light showers"
                icon = "☂"
                category = "light"
                note = (
                    "Light rainfall is estimated "
                    "for tomorrow."
                )

            else:
                label = "Heavy rain expected"
                icon = "☁"
                category = "heavy"
                note = (
                    "A higher amount of rainfall is "
                    "estimated for tomorrow."
                )

            # Custom forecast result
            st.html(f"""
            <div class="result {category}">
                <div class="result-left">
                    <div class="result-kicker">
                        TOMORROW'S FORECAST
                    </div>

                    <h3>{html.escape(label)}</h3>
                    <p>{html.escape(note)}</p>
                </div>

                <div class="result-right">
                    <span class="result-symbol">{icon}</span>
                    <div class="result-number">
                        {predicted:.2f}<small> mm</small>
                    </div>
                    <div class="result-caption">
                        Estimated rainfall
                    </div>
                </div>
            </div>
            """)

        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"Unable to generate a forecast: {exc}")

# FOOTER
st.html("""
<div class="site-footer">
    <span>Random Forest · Next-day rainfall prediction</span>
    <span>CHANGI WEATHER LAB</span>
</div>
""")