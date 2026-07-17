import streamlit as st
import pickle
import numpy as np

#Upload Model
with open("Fertilizer_Prediction.pkl",'rb') as f:
    data = pickle.load(f)

model = data['model']
le_soil = data['le_soil']
le_crop = data['le_crop']
le_fert = data['le_fert']
st.set_page_config(page_title="Fertilizer Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Fertilizer Prediction System")
st.write("Soil aur Crop ki details enter karein taake sahi khad (fertilizer) ka pata lagaya ja sake.")

#User Inpur Fields
col1, col2 = st.columns(2)
with col1:
    temp = st.number_input("Temperature (°C)", min_value=0, max_value=60, value=25)
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=30)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)

#Dropdown Value

soil_type = st.selectbox("Select Soil Type", le_soil.classes_)
crop_type = st.selectbox("Select Crop Type", le_crop.classes_)

st.subheader("Nutrients Level")
col3, col4, col5 = st.columns(3)
with col3:
    nitro = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
with col4:
    potas = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
with col5:
    phos = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=50)

# 4. Predict Button
if st.button("Predict Best Fertilizer", use_container_width=True):
    # Yahan text inputs ko model ke liye numbers mein convert kiya ja raha hai
    soil_encoded = le_soil.transform([soil_type])[0]
    crop_encoded = le_crop.transform([crop_type])[0]
    
    # Input array banayein
    input_data = np.array([[temp, humidity, moisture, soil_encoded, crop_encoded, nitro, potas, phos]])
    
    # Prediction karein (Yeh number return karega)
    prediction_encoded = model.predict(input_data)
    
    # Number ko wapas asli naam (Text) mein convert karein
    fertilizer_name = le_fert.inverse_transform(prediction_encoded)[0]
    
    st.success(f"🎉 Aapki fasal ke liye sab se behtreen fertilizer hai: **{fertilizer_name}**")

    st.success(f"🎉 Aapki fasal ke liye sab se behtreen fertilizer hai: **{fertilizer_name}**")
