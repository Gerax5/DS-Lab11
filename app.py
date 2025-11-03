import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import your modules
from modules.data_loader import load_excel_data, load_consumption_excel_data

# Color palette
COLORS = {
    'background': '#2a2e30',
    'sidebar': '#345c72',
    'accent': '#f46530',
    'secondary_accent': '#ff9e7a',
    'tertiary_accent': '#d4edf4'
}

st.set_page_config(
    page_title="Fuel Consumption & Price Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with color palette
st.markdown(f"""
    <style>
    .main {{
        padding: 0rem 1rem;
        background-color: {COLORS['background']};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['sidebar']};
    }}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("⛽ Dashboard de Consumo y precios de combustibles")

@st.cache_data
def load_data():
    # Load price data
    price_data = load_excel_data("./data/preciosLimpio.xlsx")
    
    # Load consumption data
    consumption_data = load_consumption_excel_data()
    
    return price_data, consumption_data

# Load the data
try:
    price_data, consumption_data = load_data()
except Exception as e:
    st.error(f"❌ Error cargando datos: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.header("⚙️ Configuración")

# Data type selector
data_type = st.sidebar.radio(
    "📊 Tipo de datos:",
    ["Precios", "Consumo"]
)

if data_type == "Precios":
    # Year selector for prices
    available_years = list(price_data.keys())
    selected_year = st.sidebar.selectbox("📅 Año:", available_years)
    
    # Product selector for prices
    products = ["Superior", "Regular", "Diesel", "Gas Licuado"]
    selected_products = st.sidebar.multiselect(
        "⛽ Seleccionar combustibles:",
        products,
        default=["Superior"]
    )
else:
    # Year selector for consumption
    available_years = sorted(consumption_data['Año'].unique().tolist())
    selected_year = st.sidebar.selectbox("📅 Año:", available_years)
    
    # Product selector for consumption (only Gas Licuado for now)
    selected_products = ["Gas licuado de petróleo"]

# Aggregation level
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Nivel de agregación")
aggregation = st.sidebar.radio(
    "Escala de tiempo:",
    ["Diario", "Semanal", "Mensual", "Anual"]
)

def aggregate_data(df, aggregation_level, value_columns):
    """
    Aggregate data based on selected time scale.
    
    Args:
        df: DataFrame with datetime index
        aggregation_level: "Diario", "Semanal", "Mensual", "Anual"
        value_columns: List of columns to aggregate
    
    Returns:
        Aggregated DataFrame
    """
    if aggregation_level == "Diario":
        return df
    
    df_agg = df.copy()
    
    if aggregation_level == "Semanal":
        # Resample to weekly, taking mean
        df_agg = df_agg[value_columns].resample('W').mean()
    elif aggregation_level == "Mensual":
        # Resample to monthly, taking mean
        df_agg = df_agg[value_columns].resample('M').mean()
    elif aggregation_level == "Anual":
        # Resample to yearly, taking mean
        df_agg = df_agg[value_columns].resample('Y').mean()
    
    return df_agg

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs([
    "Exploración de Datos",
    "Predicciones de modelos",
    "Comparación de modelos",
])

# =============================================================================
# TAB 1: DATA EXPLORATION
# =============================================================================
with tab1:
    if data_type == "Precios":
        # Get selected year data
        df = price_data[selected_year].copy()
        
        # Get columns that exist in the dataframe
        available_products = [p for p in selected_products if p in df.columns]
        
        if not available_products:
            st.warning("⚠️ No hay datos disponibles para los productos seleccionados")
        else:
            # Aggregate data
            df_agg = aggregate_data(df, aggregation, available_products)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📈 Precios de combustibles - {selected_year}")
                
                fig = go.Figure()
                
                for product in available_products:
                    fig.add_trace(go.Scatter(
                        x=df_agg.index,
                        y=df_agg[product],
                        mode='lines+markers' if aggregation != "Diario" else 'lines',
                        name=product,
                        line=dict(width=2),
                        marker=dict(size=6 if aggregation != "Diario" else 4),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                      'Fecha: %{x}<br>' +
                                      'Precio: Q%{y:.2f}<br>' +
                                      '<extra></extra>'
                    ))
                
                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precios (Q)",
                    hovermode='x unified',
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['tertiary_accent'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Distribución de precios")
                
                # Create histogram
                fig = go.Figure()
                
                for product in available_products:
                    fig.add_trace(go.Histogram(
                        x=df[product].dropna(),
                        name=product,
                        opacity=0.7,
                        nbinsx=30,
                        marker=dict(
                            line=dict(width=1, color='white')
                        )
                    ))
                
                fig.update_layout(
                    xaxis_title="Precio (Q)",
                    yaxis_title="Frecuencia",
                    barmode='overlay',
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['tertiary_accent'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Full width chart - Monthly averages across all years
            st.subheader("📅 Precios promedio mensuales (Todos los años)")
            
            # Combine all years data for monthly analysis
            all_data = pd.DataFrame()
            for year, df_year in price_data.items():
                df_year_copy = df_year.copy()
                df_year_copy['Year'] = year
                all_data = pd.concat([all_data, df_year_copy], ignore_index=False)
            
            # Extract month
            all_data['Month'] = all_data.index.month
            
            # Create grouped bar chart
            fig = go.Figure()
            
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            for product in available_products:
                monthly_avg = all_data.groupby('Month')[product].mean()
                fig.add_trace(go.Bar(
                    x=month_names,
                    y=monthly_avg.values,
                    name=product,
                    marker=dict(
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Mes: %{x}<br>' +
                                  'Precio Promedio: Q%{y:.2f}<br>' +
                                  '<extra></extra>'
                ))
            
            fig.update_layout(
                xaxis_title="Mes",
                yaxis_title="Precio Promedio (Q)",
                barmode='group',
                template="plotly_dark",
                height=400,
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font=dict(color=COLORS['tertiary_accent'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data view with expand option
            with st.expander("🔍 Ver datos sin procesar"):
                st.dataframe(df_agg, use_container_width=True)
    
    else:  # Consumo
        # Filter consumption data by selected year
        df_consumption = consumption_data[consumption_data['Año'] == selected_year].copy()
        
        if df_consumption.empty:
            st.warning(f"⚠️ No hay datos de consumo disponibles para el año {selected_year}")
        else:
            # Aggregate consumption data
            value_columns = ['Gas licuado de petróleo']
            df_agg = aggregate_data(df_consumption, aggregation, value_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📈 Consumo de combustible - {selected_year}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_agg.index,
                    y=df_agg['Gas licuado de petróleo'],
                    mode='lines+markers' if aggregation != "Diario" else 'lines',
                    name='Gas licuado de petróleo',
                    line=dict(width=2, color=COLORS['accent']),
                    marker=dict(size=6 if aggregation != "Diario" else 4, color=COLORS['accent']),
                    hovertemplate='<b>Gas licuado de petróleo</b><br>' +
                                  'Fecha: %{x}<br>' +
                                  'Consumo: %{y:.2f} barriles<br>' +
                                  '<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Consumo (barriles)",
                    hovermode='x unified',
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['tertiary_accent'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Distribución de consumo")
                
                # Create histogram
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=df_consumption['Gas licuado de petróleo'].dropna(),
                    name='Gas licuado de petróleo',
                    opacity=0.7,
                    nbinsx=30,
                    marker=dict(
                        color=COLORS['accent'],
                        line=dict(width=1, color='white')
                    )
                ))
                
                fig.update_layout(
                    xaxis_title="Consumo (barriles)",
                    yaxis_title="Frecuencia",
                    template="plotly_dark",
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['background'],
                    font=dict(color=COLORS['tertiary_accent']),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Full width chart - Monthly consumption across all years
            st.subheader("📅 Consumo promedio mensual (Todos los años)")
            
            # Create grouped bar chart by month
            all_consumption = consumption_data.copy()
            
            # Calculate monthly average
            monthly_avg = all_consumption.groupby('Mes')['Gas licuado de petróleo'].mean()
            
            fig = go.Figure()
            
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            fig.add_trace(go.Bar(
                x=month_names,
                y=monthly_avg.values,
                name='Gas licuado de petróleo',
                marker=dict(
                    color=COLORS['accent'],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Gas licuado de petróleo</b><br>' +
                              'Mes: %{x}<br>' +
                              'Consumo Promedio: %{y:.2f} barriles<br>' +
                              '<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Mes",
                yaxis_title="Consumo Promedio (barriles)",
                template="plotly_dark",
                height=400,
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background'],
                font=dict(color=COLORS['tertiary_accent']),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Raw data view with expand option
            with st.expander("🔍 Ver datos sin procesar"):
                st.dataframe(df_agg, use_container_width=True)

# =============================================================================
# TAB 2: MODEL PREDICTIONS
# =============================================================================
with tab2:
    st.markdown("Esta sección muestra predicciones de múltiples modelos de series temporales")
    
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
        "Seleccionar tipo de combustible para predicciones:",
        ["Gas Licuado", "Gasolina Superior"]
    )
    
    # Main prediction chart
    st.subheader(f"📈 {fuel_type} - Predicciones de consumo")
    
    # Placeholder for actual model data
    fig = go.Figure()
    
    # Add model predictions conditionally
    if show_lstm1:
        pass
    
    if show_lstm2:
        pass
    
    if show_prophet:
        pass
    
    if show_holt:
        pass
    
    if show_sarima:
        pass
    
    # Placeholder plot
    st.warning("🚧 Conecta las predicciones de tus modelos aquí. Ver el código comentado para la estructura.")
    
    fig.update_layout(
        title=f"{fuel_type} Consumo - Histórico y Predicciones",
        xaxis_title="Fecha",
        yaxis_title="Consumo (barriles)",
        hovermode='x unified',
        template="plotly_dark",
        height=500,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['tertiary_accent'])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction statistics
    st.subheader("Estadísticas de predicciones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicción de mañana", "567,234", "2.3%")
    with col2:
        st.metric("Promedio semanal", "565,120", "-0.5%")
    with col3:
        st.metric("Tendencia mensual", "↗️ Increasing", "")

# =============================================================================
# TAB 3: MODEL COMPARISON
# =============================================================================
with tab3:
    st.markdown("""
Compare diferentes modelos predictivos basándose en métricas de evaluación estándar.
Los valores más bajos indican un mejor rendimiento en las métricas de error (MAE, MSE, RMSE, AIC, BIC).
    """)
    
    # Model metrics
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

    st.subheader("📑 Tabla comparativa (según métrica seleccionada)")

    comp_table = (
        df_metrics[['Model', metric_to_plot]]
        .dropna()
        .sort_values(by=metric_to_plot, ascending=True)
        .reset_index(drop=True)
        .rename(columns={metric_to_plot: f'{metric_to_plot}'})
    )

    comp_table.insert(0, 'Rank', comp_table.index + 1)

    m_min = comp_table[f'{metric_to_plot}'].min()
    m_max = comp_table[f'{metric_to_plot}'].max()
    if m_max != m_min:
        comp_table['Score (0-100)'] = ((m_max - comp_table[f'{metric_to_plot}']) / (m_max - m_min) * 100).round(2)
    else:
        comp_table['Score (0-100)'] = 100.0

    best_val = comp_table[f'{metric_to_plot}'].iloc[0]
    worst_val = comp_table[f'{metric_to_plot}'].iloc[-1]

    comp_table['Δ vs mejor (%)'] = ((comp_table[f'{metric_to_plot}'] / best_val - 1) * 100).round(2)
    comp_table['Δ vs peor (%)']  = ((comp_table[f'{metric_to_plot}'] / worst_val - 1) * 100).round(2)

    comp_table['Etiqueta'] = ['Mejor'] + [''] * (len(comp_table) - 1)

    top_n = st.slider("Mostrar top N", min_value=1, max_value=len(comp_table), value=len(comp_table), key="top_n_comp")
    comp_table_display = comp_table.head(top_n)

    styled = (
        comp_table_display.style
        .highlight_min(subset=[f'{metric_to_plot}'], color='lightgreen')
        .highlight_max(subset=['Δ vs peor (%)'], color='lightgreen')  # el peor (más alto) queda rosado
        .bar(subset=['Score (0-100)'], vmin=0, vmax=100)
        .format({
            f'{metric_to_plot}': '{:,.2f}',
            'Score (0-100)': '{:.2f}',
            'Δ vs mejor (%)': '{:+.2f}%',
            'Δ vs peor (%)': '{:+.2f}%'
        })
    )

    st.dataframe(styled, use_container_width=True)


    st.subheader("📈 Gráficos (según métrica seleccionada)")
    # Create bar chart for selected metric
    fig = go.Figure()
    
    metric_data = df_metrics[['Model', metric_to_plot]].dropna()
    
    fig.add_trace(go.Bar(
        x=metric_data['Model'],
        y=metric_data[metric_to_plot],
        text=metric_data[metric_to_plot].round(2),
        textposition='auto',
        marker_color=COLORS['accent']
    ))
    
    fig.update_layout(
        title=f"{metric_to_plot} Comparación entre modelos",
        xaxis_title="Modelo",
        yaxis_title=metric_to_plot,
        template="plotly_dark",
        height=400,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['tertiary_accent'])
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
    
    colors_list = [COLORS['accent'], COLORS['secondary_accent'], COLORS['tertiary_accent'], '#72a0c1', '#a8dadc']
    
    for idx, row in df_normalized.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['MAE'], row['RMSE'], row['AIC'], row['BIC']],
            theta=['MAE', 'RMSE', 'AIC', 'BIC'],
            fill='toself',
            name=row['Model'],
            line=dict(color=colors_list[idx % len(colors_list)])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor=COLORS['background']
        ),
        showlegend=True,
        template="plotly_dark",
        height=500,
        title="Comparación de rendimiento (Más a los extremos es mejor)",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['tertiary_accent'])
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