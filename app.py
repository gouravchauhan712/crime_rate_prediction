import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# 🔹 Load Model & Data
# ==============================
reg_model = joblib.load("crime_rate_best_model.pkl")
city_mean_rate = joblib.load("city_mean_rate.pkl")
pop_df = pd.read_csv("population.csv")

# ==============================
# 🎨 Page Config
# ==============================
st.set_page_config(page_title="Crime Rate Predictor", page_icon="🚨", layout="centered")

# --- Custom Light CSS ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #e9f2fb, #f7fbff);
            color: #1b1b1b;
        }
        .main-card {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 25px;
            margin-top: 20px;
            color: #1b1b1b;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .title {
            text-align: center;
            color: #003366;
            font-size: 36px;
            font-weight: 700;
        }
        .sub {
            text-align: center;
            color: #4a6fa5;
            font-size: 18px;
            margin-bottom: 25px;
        }
        .metric-box {
            background-color: #f0f7ff;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            color: #003366;
            border: 1px solid #d1e3f8;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🏙️ Header
# ==============================
st.markdown("<h1 class='title'>🚨 Crime Rate Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>AI-powered prediction based on crime trends and population statistics</p>", unsafe_allow_html=True)

# ==============================
# 🧾 User Inputs
# ==============================
with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("🏙️ Select City", sorted(pop_df['City'].unique()))
        year = st.selectbox("📅 Select Year", list(range(2000, 2036)))
    with col2:
        crime_domain = st.selectbox("⚖️ Select Crime Domain", [
            "Theft", "Assault", "Murder", "Fraud", "Cybercrime", "Kidnapping", "Domestic Violence", "Robbery", "Burglary"
        ])
        month = st.selectbox("🗓️ Select Month", list(range(1, 13)))

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 🔮 Prediction
# ==============================
if st.button("🔍 Predict Crime Rate", use_container_width=True):
    mean_rate = city_mean_rate.get(city, np.mean(list(city_mean_rate.values())))

    input_df = pd.DataFrame([{
        "City": city,
        "Crime Domain": crime_domain,
        "Month": month,
        "Year": year,
        "City_MeanRate": mean_rate
    }])

    # Predict
    pred_rate = reg_model.predict(input_df)[0]

    # Population lookup
    pop = pop_df.loc[pop_df["City"] == city, "Population"]
    population = pop.values[0] if not pop.empty else pop_df["Population"].mean()
    est_cases = (pred_rate * population) / 100000

    # Determine crime level
    if pred_rate > 15:
        level = "🔴 High Crime Area"
        color = "red"
    elif pred_rate > 7:
        level = "🟠 Moderate Crime Area"
        color = "orange"
    else:
        level = "🟢 Low Crime Area"
        color = "green"

    # ==============================
    # 📊 Display Results
    # ==============================
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Results")
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric-box'><h3>{pred_rate:.2f}</h3><p>Crime Rate<br>(per 100,000)</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><h3>{int(est_cases):,}</h3><p>Estimated Cases</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><h3 style='color:{color};'>{level}</h3><p>Area Safety Level</p></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


