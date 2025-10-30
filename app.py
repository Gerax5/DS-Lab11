import streamlit as st
from modules.data_loader import load_excel_data
from modules.visualization import plot_price_distribution
from modules.ui_components import select_year, show_dataframe

# st.set_page_config(page_title="Análisis de Precios", layout="wide")

# st.title("📊 Análisis de Precios por Año")

# # Cargar datos
# data_path = "./data/preciosLimpio.xlsx"
# resultados = load_excel_data(data_path)

# # Seleccionar año
# selected_year = select_year(resultados)
# df = resultados[selected_year]

# st.subheader(f"Datos del año {selected_year}")
# show_dataframe(df)

# # Mostrar gráfica
# st.subheader("Gráfica de ejemplo")
# fig = plot_price_distribution(df, selected_year)
# st.pyplot(fig)

# st.info("💡 Puedes agregar más visualizaciones o análisis en el módulo `visualization.py`.")import streamlit as st
import pandas as pd

# --- Carga de los dataframes ---
# Asegúrate de tener estos archivos en la misma carpeta o indica la ruta correcta
# precioslimpios = pd.read_excel("./data/preciosLimpio.xlsx")
# consumos = pd.read_csv("consumos.csv")

# --- Creación de pestañas ---
tab_precios, tab_consumos = st.tabs(["Precios", "Consumos"])

# --- Pestaña: Precios ---
with tab_precios:
    st.title("Visualización de Precios")
    st.write(
        """
        En esta sección se presentan los datos procesados del conjunto *precioslimpios*,
        listos para su análisis y visualización.
        """
    )

    data_path = "./data/preciosLimpio.xlsx"
    resultados = load_excel_data(data_path)

    # Contenedor o columna completa para mantenerlo alineado
    col1, _ = st.columns([1, 3])  # más angosto el de la izquierda
    with col1:
        selected_year = select_year(resultados)

    df = resultados[selected_year]

    st.subheader(f"Datos del año {selected_year}")
    show_dataframe(df)

    df_final = pd.DataFrame()

    for nombre_hoja, df in resultados.items():
        df["Año"] = nombre_hoja
        df_final = pd.concat([df_final, df], ignore_index=True)

    st.title("Distribucion de precios")
    fig = plot_price_distribution(df_final)
    st.pyplot(fig)

# --- Pestaña: Consumos ---
with tab_consumos:
    st.title("Visualización de Consumos")
    st.write(
        """
        Aquí podrás visualizar los datos relacionados con los consumos.
        (Agrega aquí tus gráficos o tablas correspondientes.)
        """
    )
    # st.dataframe(consumos)


