# ☁️ Changi Weather Forecast Prediction (ML + Streamlit)

This project is a machine learning-based weather forecasting system that predicts next-day rainfall using historical weather data from Changi.

It combines data preprocessing, model training, and a Streamlit web application to provide a simple and interactive prediction tool.

---

## 📌 Project Goal

The main goal of this project is to predict tomorrow’s rainfall amount using today’s weather conditions such as temperature and wind speed.

Instead of treating this as a classification problem, the model predicts actual rainfall values (regression), and then converts them into simple categories for better user understanding.

---

## 🧠 How the System Works

The workflow of the project is as follows:

1. A historical dataset (`changi.csv`) is loaded  
2. A new target column is created by shifting rainfall values to represent next-day rainfall  
3. Feature selection is performed  
4. A Random Forest Regressor model is trained  
5. The trained model and feature list are saved using `joblib`  
6. A Streamlit app loads the saved model  
7. User enters weather values through the UI  
8. The model predicts rainfall amount  
9. The result is displayed in a simple and readable format  

---

## 📊 Features Used for Prediction

The model uses only daily weather conditions that are available at prediction time:

- Month  
- Mean Temperature (°C)  
- Maximum Temperature (°C)  
- Minimum Temperature (°C)  
- Mean Wind Speed (km/h)  
- Max Wind Speed (km/h)  

These features help the model understand seasonal variation and short-term weather behavior.

---

## 🛠️ Tech Stack

This project is built using the following technologies:

```text
Python
Pandas
Scikit-learn
RandomForestRegressor
Streamlit
Joblib
