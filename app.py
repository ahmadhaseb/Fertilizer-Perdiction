import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Model load karne ka function
@st.cache_resource
def load_model():
    with open('fertilizer_model.pkl', 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'model' in data:
        return data['model']
    return data

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load karne mein masla aya: {e}")

# Page styling
st.set_page_config(page_title="Fertilizer Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Fertilizer Prediction System")
st.write("Mitti aur fasal ki details enter karein taake sahi khad (fertilizer) ka pata chal sake.")

# 2. Mappings
soil_mapping = {
    "Black (Kali Mitti)": 0,
    "Clayey (Chikni Mitti)": 1,
    "Loamy (Zarkhez / Meera Mitti)": 2,
    "Red (Surkh Mitti)": 3,
    "Sandy (Retili Mitti)": 4
}

crop_mapping = {
    "Barley (Jao)": 0,
    "Cotton (Kapaas)": 1,
    "Groundnuts (Moongphali)": 2,
    "Maize (Makai)": 3,
    "Millets (Bajra)": 4,
    "Oil seeds (Teel Dar Ajnas)": 5,
    "Paddy (Chawal / Dhan)": 6,
    "Pulses (Dalein)": 7,
    "Sugarcane (Ganna)": 8,
    "Tobacco (Tambaku)": 9,
    "Wheat (Gandum)": 10
}

fertilizer_mapping = {
    0: "10-26-26 (NPK)",
    1: "28-28 (NPK)",
    2: "14-35-14 (NPK)",
    3: "20-20 (NPK)",
    4: "DAP (Di-Ammonium Phosphate)",
    5: "MOP (Muriate of Potash)",
    6: "Urea"
}

# 3. User Input Fields
col1, col2 = st.columns(2)
with col1:
    temp = st.number_input("Temperature (°C)", min_value=0, max_value=60, value=25)
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=30)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)

soil_selected = st.selectbox("Select Soil Type", list(soil_mapping.keys()))
crop_selected = st.selectbox("Select Crop Type", list(crop_mapping.keys()))

st.subheader("Nutrients Level")
col3, col4, col5 = st.columns(3)
with col3:
    nitro = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
with col4:
    potas = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
with col5:
    phos = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=50)

# 4. Prediction Button and Input Summary
if st.button("Predict Best Fertilizer", use_container_width=True):
    soil_encoded = soil_mapping[soil_selected]
    crop_encoded = crop_mapping[crop_selected]
    
    input_data = np.array([[temp, humidity, moisture, soil_encoded, crop_encoded, nitro, potas, phos]])
    
    try:
        # Array se pehla element nikalne ke liye [0] lagaya hai
        prediction_array = model.predict(input_data)
        prediction_number = int(prediction_array[0]) 
        
        fertilizer_name = fertilizer_mapping.get(prediction_number, f"Unknown (Code: {prediction_number})")
        
        # 1. Prediction Result Display
        st.success(f"🎉 Aapki fasal ke liye sab se behtreen fertilizer hai: **{fertilizer_name}**")
        
        # 2. User Input Summary Display
        st.write("---")
        st.subheader("📋 Aap Ka Diya Hua Data (Input Summary)")
        
        summary_df = pd.DataFrame({
            "Parameters (Features)": [
                "Temperature", "Humidity", "Moisture", 
                "Soil Type", "Crop Type", 
                "Nitrogen (N)", "Potassium (K)", "Phosphorous (P)"
            ],
            "Aap Ki Values": [
                f"{temp} °C", f"{humidity} %", f"{moisture} %", 
                soil_selected, crop_selected, 
                f"{nitro}", f"{potas}", f"{phos}"
            ]
        })
        
        st.table(summary_df.set_index("Parameters (Features)"))
        
    except Exception as e:
        st.error(f"Prediction ke dauran error aya: {e}")
