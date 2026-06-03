import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model

model = joblib.load("predictive_model.pkl")

# App title

st.title("Predictive Maintenance System")

st.write(
"Predict machine failures using industrial IoT sensor data"
)

# Inputs

air_temp = st.number_input(
"Air Temperature (K)",
value=300
)

process_temp = st.number_input(
"Process Temperature (K)",
value=310
)

rpm = st.number_input(
"Rotational Speed (RPM)",
value=1500
)

torque = st.number_input(
"Torque (Nm)",
value=40
)

tool_wear = st.number_input(
"Tool Wear (min)",
value=20
)

# Machine type

machine_type = st.selectbox(
"Machine Type",
["L", "M", "H"]
)

# Encode machine type

type_L = 0
type_M = 0

if machine_type == "L":
type_L = 1

elif machine_type == "M":
type_M = 1

# Prediction

if st.button("Predict Failure"):

```
sample = pd.DataFrame([[
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    type_L,
    type_M
]],
columns=[
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
    "Type_L",
    "Type_M"
])

prediction = model.predict(sample)

probability = model.predict_proba(sample)

failure_probability = probability[0][1]

# Result

st.subheader("Prediction Result")

if prediction[0] == 1:

    st.error(
        f"High Failure Risk: {failure_probability:.2%}"
    )

else:

    st.success(
        f"Machine Healthy: {(1-failure_probability):.2%}"
    )

# Probability chart

prob_df = pd.DataFrame({
    "Status": [
        "Healthy",
        "Failure"
    ],
    "Probability": [
        1 - failure_probability,
        failure_probability
    ]
})

fig = px.bar(
    prob_df,
    x="Status",
    y="Probability",
    title="Machine Failure Probability"
)

st.plotly_chart(fig)
