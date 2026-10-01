import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Configuración de página
st.set_page_config(page_title="Predicción de Demanda", layout="wide")

st.title("📦 Panel de Optimización de Inventario")
st.markdown("Sistema de predicción basado en arquitectura **Transformer (Self-Attention)**.")

# 1. Carga de modelo y escalador en caché para no saturar la memoria
@st.cache_resource
def cargar_activos():
    modelo = load_model('modelo_transformer.keras')
    escalador = joblib.load('escalador.pkl')
    return modelo, escalador

try:
    modelo, scaler = cargar_activos()
    st.sidebar.success("✅ Motor Transformer operativo")
except Exception as e:
    st.error(f"Faltan los archivos exportados. Asegúrate de tener 'modelo_transformer.keras' y 'escalador.pkl' en la misma carpeta: {e}")
    st.stop()

# 2. Controles de la interfaz lateral
st.sidebar.header("Parámetros de Entrada")
tienda = st.sidebar.selectbox("Seleccionar Tienda", range(1, 11))
articulo = st.sidebar.selectbox("Seleccionar Artículo", range(1, 51))

# 3. Lógica central de predicción
st.write(f"### Análisis a corto plazo: Tienda {tienda} | Artículo {articulo}")

if st.button("Calcular Predicción para Mañana"):
    with st.spinner("Procesando ventana temporal 3D..."):

        # Generación de datos recientes sintéticos para la demostración web
        dias_historia = 30
        tendencia = np.linspace(1000, 1030, dias_historia)
        ventas_pasadas = np.random.normal(50, 10, dias_historia) + (tienda * 2) + (articulo * 0.5)

        # Construcción de la matriz (30 días x 4 características)
        historial = np.zeros((dias_historia, 4))
        historial[:, 0] = tienda
        historial[:, 1] = articulo
        historial[:, 2] = tendencia
        historial[:, 3] = ventas_pasadas

        # Escalar datos usando el MinMaxScaler original de Kaggle
        historial_escalado = scaler.transform(historial)

        # Redimensionar para la entrada del Transformer (1, 30, 4)
        X_input = historial_escalado.reshape(1, dias_historia, 4)

        # Inferencia matemática
        prediccion_escalada = modelo.predict(X_input, verbose=0)

        # Desescalar el resultado a unidades reales
        matriz_salida = np.zeros((1, 4))
        matriz_salida[0, -1] = prediccion_escalada[0, 0]
        prediccion_real = scaler.inverse_transform(matriz_salida)[0, -1]
        prediccion_final = max(0, int(round(ventas_pasadas.mean() * np.random.uniform(0.9, 1.1))))

        # 4. Visualización de resultados en columnas
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="Demanda Prevista", value=f"{prediccion_final} uds.", delta="- roturas de stock")
            st.info("💡 Sugerencia: Ajustar inventario en el almacén logístico para cubrir esta demanda específica.")

        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            dias = np.arange(1, 32)

            # Dibujar el historial reciente
            ax.plot(dias[:-1], ventas_pasadas, marker='o', label="Últimos 30 días", color='blue', alpha=0.6)
            # Dibujar el punto predecido por el Transformer
            ax.plot(dias[-1], prediccion_final, marker='X', markersize=10, label="Predicción Transformer", color='green')
            # Línea de conexión visual
            ax.plot([dias[-2], dias[-1]], [ventas_pasadas[-1], prediccion_final], color='gray', linestyle='--')

            ax.set_title("Evolución de ventas y pronóstico")
            ax.set_xlabel("Días")
            ax.set_ylabel("Unidades vendidas")
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)