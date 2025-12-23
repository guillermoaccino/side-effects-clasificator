import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo y las herramientas
# Usamos @st.cache_resource para que no lo recargue cada vez que tocas un botón
@st.cache_resource
def cargar_modelo():
    with open('modelo_droga_final.bin', 'rb') as f:
        datos = pickle.load(f)
    return datos

datos = cargar_modelo()
model = datos['modelo']
dv = datos['vectorizer']
scaler = datos['scaler']
le = datos['encoder']

# Título y Descripción de la App
st.title("💊 Predicción de Efectos Secundarios")
st.markdown("""
Esta aplicación utiliza Inteligencia Artificial para predecir posibles efectos adversos 
en pacientes basándose en su perfil clínico y el tratamiento administrado.
""")

# Formulario de entrada de datos 
st.sidebar.header("Datos del Paciente")

def obtener_datos_usuario():
    lista_drogas = ['Aspirin', 'Amoxicillin', 'Ibuprofen', 'Paracetamol', 'Lipitor'] 
    lista_condiciones = ['Pain', 'Infection', 'High Blood Pressure', 'Anxiety']

    age = st.sidebar.slider("Edad", 0, 100, 30)
    gender = st.sidebar.selectbox("Género", ["Male", "Female"])
    condition = st.sidebar.selectbox("Condición Médica", lista_condiciones)
    drug_name = st.sidebar.selectbox("Medicamento", lista_drogas)
    dosage = st.sidebar.number_input("Dosis (mg)", min_value=0.0, value=500.0)
    duration = st.sidebar.number_input("Duración (días)", min_value=1, value=7)

    #Variables de entrada
    data = {
        'Age': age,
        'Gender': gender,
        'Condition': condition,
        'Drug_Name': drug_name,
        'Dosage_mg': dosage,
        'Treatment_Duration_Days': duration
    }
    return data

input_data = obtener_datos_usuario()

st.subheader("Resumen del Paciente:")
st.json(input_data)

if st.button("🔍 Predecir Efecto Secundario"):

    X_vectorized = dv.transform([input_data])

    X_scaled = scaler.transform(X_vectorized)

    prediction_num = model.predict(X_scaled)

    prediction_text = le.inverse_transform(prediction_num)

    st.success(f"⚠️ Efecto Secundario Predicho: **{prediction_text[0]}**")
 
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)
        st.write("Confianza del modelo:", prob.max())