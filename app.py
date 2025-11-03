import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import your modules
from modules.data_loader import load_excel_data

st.set_page_config(
    page_title="Fuel Consumption & Price Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("⛽ Dashboard de Consumo y precios de combustibles")

@st.cache_data
def load_data():
    # Load price data
    price_data = load_excel_data("./data/preciosLimpio.xlsx")
    print(price_data)
    
    # Load consumption data
    # consumption_data = pd.read_excel("./data/CONSUMO-HIDROCARBUROS-2024-12.xlsx", header=6)
    
    return price_data

# Load the data
try:
    price_data = load_data()
    # st.sidebar.success("✅ Data loaded successfully")
except Exception as e:
    # st.sidebar.error(f"❌ Error loading data: {e}")
    st.stop()

# Year selector in sidebar
available_years = list(price_data.keys())
selected_year = st.sidebar.selectbox("📅 Año:", available_years)

# Product selector
products = ["Superior", "Regular", "Diésel", "Gas Licuado"]
selected_products = st.sidebar.multiselect(
    "⛽ Select Fuel Types:",
    products,
    default=["Superior"]
)

# Date range selector
st.sidebar.markdown("---")
st.sidebar.subheader("Nivel de agregación")
aggregation = st.sidebar.radio(
    "Escala de tiempo:",
    ["Daily", "Weekly", "Monthly", "Yearly"]
)

# Create tabs for different sections
tab1, tab2, tab3= st.tabs([
    "Exploración de Datos",
    "Predicciones de modelos",
    "Comparación de modelos",
])

# =============================================================================
# TAB 1: DATA EXPLORATION
# =============================================================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        
        # Get selected year data
        df = price_data[selected_year]
        
        fig = go.Figure()
        
        for product in selected_products:
            if product in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[product],
                    mode='lines',
                    name=product,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Fecha: %{x}<br>' +
                                  'Precio: Q%{y:.2f}<br>' +
                                  '<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"Precios de combustibles - {selected_year}",
            xaxis_title="Fecha",
            yaxis_title="Precios (Q)",
            hovermode='x unified',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        
        # Create histogram
        fig = go.Figure()
        
        for product in selected_products:
            if product in df.columns:
                fig.add_trace(go.Histogram(
                    x=df[product].dropna(),
                    name=product,
                    opacity=0.7,
                    nbinsx=30
                ))
        
        fig.update_layout(
            title="Distribución de precios",
            xaxis_title="Precio (Q)",
            yaxis_title="Frecuencia",
            barmode='overlay',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Full width charts
    st.subheader("📅 Precios promedio")
    
    # Combine all years data for monthly analysis
    all_data = pd.DataFrame()
    for year, df_year in price_data.items():
        df_year_copy = df_year.copy()
        df_year_copy['Year'] = year
        all_data = pd.concat([all_data, df_year_copy], ignore_index=True)
    
    # Extract month
    if not pd.api.types.is_datetime64_any_dtype(all_data.index):
        all_data['Date'] = pd.to_datetime(all_data.index, errors='coerce')
    else:
        all_data['Date'] = all_data.index
    
    all_data['Month'] = all_data['Date'].dt.month
    
    # Create grouped bar chart
    fig = go.Figure()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for product in selected_products:
        if product in all_data.columns:
            monthly_avg = all_data.groupby('Month')[product].mean()
            fig.add_trace(go.Bar(
                x=month_names,
                y=monthly_avg.values,
                name=product,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Month: %{x}<br>' +
                              'Avg Price: Q%{y:.2f}<br>' +
                              '<extra></extra>'
            ))
    
    fig.update_layout(
        title="Average Monthly Prices (All Years)",
        xaxis_title="Month",
        yaxis_title="Average Price (Q)",
        barmode='group',
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data view with expand option
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df, use_container_width=True)

# =============================================================================
# TAB 2: MODEL PREDICTIONS
# =============================================================================
with tab2:
    st.info("📌 This section displays predictions from multiple time series models")
    
    # Model selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_lstm1 = st.checkbox("LSTM Model 1", value=True)
    with col2:
        show_lstm2 = st.checkbox("LSTM Model 2 (Tuned)", value=True)
    with col3:
        show_prophet = st.checkbox("Prophet Model", value=False)
    
    col4, col5 = st.columns(2)
    with col4:
        show_holt = st.checkbox("Holt-Winters Model", value=False)
    with col5:
        show_sarima = st.checkbox("SARIMA Model", value=False)
    
    st.markdown("---")
    
    # Fuel type selector for predictions
    fuel_type = st.selectbox(
        "Select fuel type for predictions:",
        ["Gas Licuado", "Gasolina Superior"]
    )
    
    # Main prediction chart
    st.subheader(f"📈 {fuel_type} - Consumption Predictions")
    
    # Placeholder for actual model data
    # You'll replace this with your actual model predictions
    fig = go.Figure()
    
    # Example: Add actual consumption data (you'll load your real data here)
    # dates = pd.date_range(start='2000-01-01', end='2024-12-31', freq='M')
    # actual = np.random.randn(len(dates)).cumsum() + 300000
    
    # fig.add_trace(go.Scatter(
    #     x=dates,
    #     y=actual,
    #     mode='lines',
    #     name='Actual Consumption',
    #     line=dict(color='white', width=2)
    # ))
    
    # Add model predictions conditionally
    if show_lstm1:
        # Add LSTM Model 1 predictions
        # fig.add_trace(go.Scatter(
        #     x=prediction_dates,
        #     y=lstm1_predictions,
        #     mode='lines',
        #     name='LSTM Model 1',
        #     line=dict(color='#00ff00', dash='dash')
        # ))
        pass
    
    if show_lstm2:
        # Add LSTM Model 2 predictions
        pass
    
    if show_prophet:
        # Add Prophet predictions
        pass
    
    if show_holt:
        # Add Holt-Winters predictions
        pass
    
    if show_sarima:
        # Add SARIMA predictions
        pass
    
    # Placeholder plot
    st.warning("🚧 Connect your model predictions here. See the commented code for structure.")
    
    fig.update_layout(
        title=f"{fuel_type} Consumption - Historical & Predicted",
        xaxis_title="Date",
        yaxis_title="Consumption (barrels)",
        hovermode='x unified',
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Zoom into predictions only
    st.subheader("🔍 Prediction Period Detail")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    with col2:
        st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    
    # Create zoomed prediction chart
    fig_zoom = go.Figure()
    
    # Add predictions only (you'll populate this with real data)
    
    fig_zoom.update_layout(
        title="Zoomed Prediction Comparison",
        xaxis_title="Date",
        yaxis_title="Consumption (barrels)",
        hovermode='x unified',
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig_zoom, use_container_width=True)
    
    # Prediction statistics
    st.subheader("📊 Prediction Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Next Day Prediction", "567,234", "2.3%")
    with col2:
        st.metric("7-Day Average", "565,120", "-0.5%")
    with col3:
        st.metric("30-Day Trend", "↗️ Increasing", "")

# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================
with tab3:
    st.markdown("""
Compare diferentes modelos predictivos basándose en métricas de evaluación estándar.
Los valores más bajos indican un mejor rendimiento en las métricas de error (MAE, MSE, RMSE, AIC, BIC).
    """)
    
    # Model metrics (you'll populate this with your actual metrics)
    metrics_gas_licuado = {
        'Model': ['LSTM 1', 'LSTM 2 (Tuned)', 'Holt-Winters', 'Prophet', 'SARIMA'],
        'MAE': [30240.46, 32097.82, 24936.78, 19578.26, None],
        'MSE': [1.562e9, 1.623e9, None, None, None],
        'RMSE': [39525.16, 42768.87, None, None, None],
        'AIC': [1276.99, 1286.30, 617.61, 617.31, 5549.75],
        'BIC': [1306.08, 1315.39, 631.28, 637.82, 5574.32]
    }
    
    metrics_gasolina = {
        'Model': ['LSTM 1', 'LSTM 2 (Tuned)', 'Holt-Winters', 'Prophet', 'SARIMA'],
        'MAE': [53370.53, 61069.12, 73433.39, 46745.25, None],
        'MSE': [5.498e9, 5.652e9, None, None, None],
        'RMSE': [66821.48, 72786.50, None, None, None],
        'AIC': [1361.17, 1545.43, 677.07, 669.09, 5861.61],
        'BIC': [1390.49, 1760.84, 690.75, 689.59, 5879.18]
    }
    
    fuel_type_comparison = st.selectbox(
        "Tipo de combustible:",
        ["Gas Licuado", "Gasolina Superior"],
        key="comparison_fuel"
    )
    
    if fuel_type_comparison == "Gas Licuado":
        df_metrics = pd.DataFrame(metrics_gas_licuado)
    else:
        df_metrics = pd.DataFrame(metrics_gasolina)
    
    # Display metrics table
    st.subheader("📋 Métricas de rendimiento")
    st.dataframe(
        df_metrics.style.highlight_min(axis=0, subset=['MAE', 'RMSE', 'AIC', 'BIC'], color='lightgreen'),
        use_container_width=True
    )
    
    # Visualize metrics comparison
    st.subheader("📊 Comparación visual")
    
    metric_to_plot = st.selectbox(
        "Métrica a comparar:",
        ["MAE", "RMSE", "AIC", "BIC"]
    )
    
    # Create bar chart for selected metric
    fig = go.Figure()
    
    metric_data = df_metrics[['Model', metric_to_plot]].dropna()
    
    fig.add_trace(go.Bar(
        x=metric_data['Model'],
        y=metric_data[metric_to_plot],
        text=metric_data[metric_to_plot].round(2),
        textposition='auto',
        marker_color='#f46530'
    ))
    
    fig.update_layout(
        title=f"{metric_to_plot} Comparación entre modelos",
        xaxis_title="Modelo",
        yaxis_title=metric_to_plot,
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-metric radar chart
    st.subheader("🎯 Comparación multimodelo")
    
    # Normalize metrics for radar chart (0-1 scale, inverted so higher is better)
    df_normalized = df_metrics.copy()
    
    for col in ['MAE', 'RMSE', 'AIC', 'BIC']:
        if col in df_normalized.columns:
            max_val = df_normalized[col].max()
            min_val = df_normalized[col].min()
            if max_val != min_val:
                # Invert: lower values become higher scores
                df_normalized[col] = 1 - ((df_normalized[col] - min_val) / (max_val - min_val))
    
    fig = go.Figure()
    
    for idx, row in df_normalized.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['MAE'], row['RMSE'], row['AIC'], row['BIC']],
            theta=['MAE', 'RMSE', 'AIC', 'BIC'],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        template="plotly_dark",
        height=500,
        title="Comparación de rendimiento (Más a los extremos es mejor)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model recommendation
    st.subheader("Conclusiones de modelos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Mejor modelo en general: Prophet**")
        st.markdown("""
        - Valores AIC y BIC más bajos
        - Mejor equilibrio entre precisión y complejidad
        - Buena generalización
        """)
    
    with col2:
        st.info("**Rendimiento LSTM**")
        st.markdown("""
        - Mejor que SARIMA
        - Competitivo con los métodos tradicionales
        - Posibilidad de mejora con más datos
        """)