import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

#Load Data
df = pd.read_csv("changi.csv")

#Shift Target (Predict tomorrow's rain using today's data)
df['Tomorrow_Rainfall'] = df['Daily Rainfall Total (mm)'].shift(-1)

#Select Clean Features
#We use features,would actually know at the end of the day
features = [
    'Month', 'Mean Temperature (°C)', 'Maximum Temperature (°C)', 
    'Minimum Temperature (°C)', 'Mean Wind Speed (km/h)', 'Max Wind Speed (km/h)'
]

df = df.dropna(subset=['Tomorrow_Rainfall'])

X = df[features]
Y = df['Tomorrow_Rainfall']

#Train Model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)  

#Save Model and Feature List
joblib.dump(model, "weather_model.pkl")
joblib.dump(features, "weather_features.pkl")

print("Training Complete. features.pkl and model.pkl created.")