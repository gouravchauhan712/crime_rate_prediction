# ==============================
# 📦 Import Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ==============================
# 📂 Load Dataset
# ==============================
crime_df = pd.read_csv("D:\crime_rate_prediction\crime_dataset_india.csv")   # Replace with your path
pop_df = pd.read_csv("D:\crime_rate_prediction\population.csv")               # Columns: City, Population

# ==============================
# 🧹 Data Cleaning & Preparation
# ==============================
crime_df['Date Reported'] = pd.to_datetime(crime_df['Date Reported'], errors='coerce')

# Extract Year and Month
crime_df['Month'] = crime_df['Date Reported'].dt.month
crime_df['Year'] = crime_df['Date Reported'].dt.year

# Drop rows with missing critical info
crime_df.dropna(subset=['City', 'Crime Domain', 'Month', 'Year'], inplace=True)

# Merge with population data
data = crime_df.merge(pop_df, on='City', how='left')
data['Population'] = data['Population'].fillna(data['Population'].mean())

# Compute total crimes per City & Domain
crime_counts = data.groupby(['City', 'Crime Domain']).size().reset_index(name='Crime_Count')
data = data.merge(crime_counts, on=['City', 'Crime Domain'], how='left')

# Compute Crime Rate per 100,000 people
data['Crime_Rate'] = (data['Crime_Count'] / data['Population']) * 100000

# Drop rows with invalid or missing data
data = data.dropna(subset=['Crime_Rate'])

# ==============================
# 🔢 Feature Engineering
# ==============================
# Mean encoding for City
city_mean_rate = data.groupby('City')['Crime_Rate'].mean().to_dict()
data['City_MeanRate'] = data['City'].map(city_mean_rate)

# Select features and target
features = ['City', 'Crime Domain', 'Month', 'Year', 'City_MeanRate']
target = 'Crime_Rate'

# ==============================
# 🔄 Preprocessing
# ==============================
cat_features = ['City', 'Crime Domain']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# ==============================
# 📊 Train/Test Split
# ==============================
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 🧠 Define Models
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
}

results = {}

# ==============================
# 🏋️ Train & Evaluate Models
# ==============================
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"model": pipe, "RMSE": rmse, "R2": r2}
    print(f"🔎 {name} -- RMSE: {rmse:.2f}, R²: {r2:.2f}")

# ==============================
# 🏆 Select Best Model
# ==============================
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = results[best_model_name]['model']
best_rmse = results[best_model_name]['RMSE']
best_r2 = results[best_model_name]['R2']

print("\n🏆 Best Model:", best_model_name)
print(f"✅ RMSE: {best_rmse:.2f}, R²: {best_r2:.2f}")

# ==============================
# 💾 Save Model
# ==============================
joblib.dump(best_model, "crime_rate_best_model.pkl")
joblib.dump(city_mean_rate, "city_mean_rate.pkl")
print("📁 Best model and city mean rate mapping saved successfully!")

# ==============================
# 🔍 Example Prediction
# ==============================
sample_city = "Delhi"
sample_domain = "Theft"
sample_month = 5
sample_year = 2023

sample_input = pd.DataFrame([{
    "City": sample_city,
    "Crime Domain": sample_domain,
    "Month": sample_month,
    "Year": sample_year,
    "City_MeanRate": city_mean_rate.get(sample_city, np.mean(list(city_mean_rate.values())))
}])

pred = best_model.predict(sample_input)[0]
print(f"\n🚨 Predicted Crime Rate for {sample_city} ({sample_domain}, {sample_month}/{sample_year}): {pred:.2f} per 100,000 population")
