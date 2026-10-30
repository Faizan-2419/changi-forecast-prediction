# ☁️ Changi Atmosphere - Weather Forecast Prediction

### Machine Learning | Python | Streamlit

An interactive, machine learning-powered weather forecasting application designed to predict **next-day rainfall in millimeters (mm)** using historical weather observations from Changi, Singapore.

The project combines data preprocessing, Random Forest regression, and a modern Streamlit interface to transform weather observations into accessible rainfall estimates.

## 📌 Project Overview

Changi Atmosphere is designed to estimate tomorrow's rainfall based on today's temperature, wind speed, and seasonal information.

The system uses a **Random Forest Regressor** to generate a numerical rainfall prediction, which is then translated into descriptive weather categories for easier interpretation.

## ✨ Key Features

- **Next-Day Rainfall Prediction:** Estimates rainfall in millimeters using machine learning.
- **Interactive Weather Inputs:** Allows users to enter temperature, wind speed, and month.
- **Rainfall Classification:** Displays descriptive rainfall categories based on predicted values.
- **Input Validation:** Checks temperature and wind-speed consistency.
- **Modern Web Interface:** Responsive Streamlit application with custom HTML and CSS.
- **Model Integration:** Loads a trained machine learning model using Joblib.

## 🧠 How It Works

1. Load historical weather observations from `changi.csv`.
2. Prepare the dataset and create the next-day rainfall target using `shift(-1)`.
3. Select six weather features for model training.
4. Split the dataset into training and testing subsets.
5. Train a Random Forest regression model using Scikit-learn.
6. Save the trained model and selected feature list using Joblib.
7. Collect weather observations through the Streamlit interface.
8. Generate and display the predicted rainfall amount and corresponding weather category.

## 📊 Input Features

The model uses six weather parameters:

| Feature | Unit |
|---|---|
| Month | 1–12 |
| Mean Temperature | °C |
| Maximum Temperature | °C |
| Minimum Temperature | °C |
| Mean Wind Speed | km/h |
| Maximum Wind Speed | km/h |

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and machine learning |
| Pandas | Data preprocessing and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Random Forest regression |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web application |
| HTML & CSS | Custom interface design and styling |

## 🌐 Live Application

**Explore Changi Atmosphere:**

[Launch Weather Forecast App](https://changi-atmosphere.streamlit.app)

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Faizan-2419/changi-forecast-prediction.git
cd changi-forecast-prediction
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
streamlit run app.py
```

The application will launch in your browser.

## 📁 Project Structure

```text
changi-forecast-prediction/
│
├── app.py                    # Streamlit application
├── style.css                 # Custom UI styling
├── train.py                  # Model training
├── changi.csv                # Historical weather dataset
├── weather_model.pkl         # Trained regression model
├── weather_features.pkl      # Model input features
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🔮 Future Enhancements

- Integration of live weather data APIs
- Additional meteorological features such as humidity and atmospheric pressure
- Automated weather data collection
- Expanded historical datasets
- Advanced forecasting models and time-series techniques

---

**Developed by Faizanur Rahman**

*Changi Atmosphere — Exploring weather forecasting through machine learning.*
