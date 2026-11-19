import joblib
import pandas as pd
import streamlit as st

# 1. Page title and short intro
st.set_page_config(page_title="Daily Commute Fare Estimator", page_icon="🛵")
st.title("Daily Commute & Trip Cost Estimator")
st.write("Adjust your trip details below to estimate your fare and compare transport modes.")

# 2. Load the saved model pipeline (.pkl)
@st.cache_resource
def load_model():
    return joblib.load("commute_fare_model.pkl")

model = load_model()

# 3. User input controls
col1, col2 = st.columns(2)

with col1:
    distance_km = st.slider("Trip Distance (km)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
    hour_of_day = st.slider("Departure Hour (0–23)", min_value=0, max_value=23, value=9, step=1)

with col2:
    traffic_index = st.slider(
        "Traffic Severity (1.0 = Light, 2.5 = Heavy)",
        min_value=1.0,
        max_value=2.5,
        value=1.6,
        step=0.1,
    )
    selected_mode = st.selectbox("Primary Transport Mode", ["Transit", "Bike", "Auto", "Cab"])

# 4. Automatically compute peak hour flag (8-10 AM and 5-8 PM)
is_peak = 1 if hour_of_day in [8, 9, 10, 17, 18, 19, 20] else 0
st.caption(f"Peak Hour Status: **{'Yes (Rush Hour)' if is_peak == 1 else 'No (Off-Peak)'}**")

# 5. Predict fare for the selected mode
input_df = pd.DataFrame([
    {
        "distance_km": distance_km,
        "hour_of_day": hour_of_day,
        "is_peak": is_peak,
        "traffic_index": traffic_index,
        "mode": selected_mode,
    }
])

predicted_fare = float(model.predict(input_df)[0])
st.metric(label=f"Estimated {selected_mode} Fare", value=f"₹{predicted_fare:.2f}")

# 6. Compare all modes for the exact same trip
st.subheader("Compare All Modes (Cheapest First)")
modes_list = ["Transit", "Bike", "Auto", "Cab"]
compare_df = pd.DataFrame([
    {
        "distance_km": distance_km,
        "hour_of_day": hour_of_day,
        "is_peak": is_peak,
        "traffic_index": traffic_index,
        "mode": m,
    }
    for m in modes_list
])

compare_df["Estimated Fare (₹)"] = model.predict(compare_df).round(2)
compare_sorted = compare_df[["mode", "Estimated Fare (₹)"]].sort_values(
    by="Estimated Fare (₹)", ascending=True
)

st.dataframe(compare_sorted, hide_index=True, use_container_width=True)