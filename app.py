import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# --- CONFIGURACIÓN DE LA PÁGINA (Debe ser lo primero) ---
st.set_page_config(
    page_title="Drug Side-Effects AI",
    page_icon="💊",
    layout="centered"
)

# --- ESTILOS CSS PERSONALIZADOS (Opcional, para dar color) ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        with open('modelo_droga_final.bin', 'rb') as f:
            datos = pickle.load(f)
        return datos
    except FileNotFoundError:
        st.error("❌ No se encuentra el archivo 'modelo_droga_final.bin'. Asegúrate de que está en la misma carpeta.")
        return None

datos = cargar_modelo()

if datos:
    model = datos['modelo']
    dv = datos['vectorizer']
    le = datos['encoder']
    # Nota: He visto que has quitado el scaler. Asumo que este modelo
    # se entrenó sin escalar (Decision Tree o SVM raw). Si no, daría error.

    # --- ENCABEZADO ---
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) # Icono médico genérico
    st.title("IA: Predicción de Efectos Secundarios")
    st.markdown("""
    Esta herramienta asiste al personal médico estimando reacciones adversas mediante 
    **Machine Learning**. Introduce los datos clínicos en el menú lateral.
    """)
    st.divider()

    # --- SIDEBAR (Entrada de datos) ---
    st.sidebar.header("📋 Historial del Paciente")
    
    def obtener_datos_usuario():
        # Listas de ejemplo (Asegúrate que coinciden con tu entrenamiento)
        lista_drogas = ['Aspirin', 'Amoxicillin', 'Ibuprofen', 'Paracetamol', 'Lipitor'] 
        lista_condiciones = ['Pain', 'Infection', 'High Blood Pressure', 'Anxiety']

        with st.sidebar.form("formulario_paciente"):
            age = st.slider("Edad", 0, 100, 45)
            gender = st.selectbox("Género", ["Male", "Female"])
            condition = st.selectbox("Condición Médica", lista_condiciones)
            st.markdown("---")
            drug_name = st.selectbox("Medicamento Prescrito", lista_drogas)
            dosage = st.number_input("Dosis (mg)", min_value=0.0, value=500.0, step=50.0)
            duration = st.number_input("Duración Tratamiento (días)", min_value=1, value=7)
            
            submit_btn = st.form_submit_button("Aplicar Cambios")

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

    # --- ÁREA PRINCIPAL: RESUMEN VISUAL ---
    st.subheader("👤 Perfil del Paciente")
    
    # Usamos columnas para que quede más organizado
    col1, col2, col3 = st.columns(3)
    col1.metric("Edad", input_data['Age'])
    col2.metric("Género", input_data['Gender'])
    col3.metric("Condición", input_data['Condition'])

    col4, col5 = st.columns(2)
    col4.info(f"💊 **Fármaco:** {input_data['Drug_Name']}")
    col5.info(f"⚖️ **Dosis:** {input_data['Dosage_mg']} mg / {input_data['Treatment_Duration_Days']} días")

    st.markdown("---")

    # --- BOTÓN Y PREDICCIÓN ---
    if st.button("🔍 Analizar Riesgo", type="primary"):
        
        # Simular tiempo de cómputo (efecto visual)
        with st.spinner('Procesando perfil clínico...'):
            time.sleep(1) 
            
            X_vectorized = dv.transform([input_data])
            
            prediction_num = model.predict(X_vectorized)
            prediction_text = le.inverse_transform(prediction_num)[0]

            st.success("✅ Análisis Completado")
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                <h3 style="color: #333; margin:0;">Resultado Predicho:</h3>
                <h1 style="color: #ff4b4b; margin:0;">{prediction_text}</h1>
            </div>
            """, unsafe_allow_html=True)


else:
    st.warning("Esperando archivo del modelo...")
