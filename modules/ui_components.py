import streamlit as st

def select_year(resultados: dict) -> str:
    years = list(resultados.keys())
    selected_year = st.sidebar.selectbox("Selecciona el año:", years)
    return selected_year

def show_dataframe(df):
    st.dataframe(df, use_container_width=True)
