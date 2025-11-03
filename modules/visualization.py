import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


def create_price_time_series(df, products, title="Fuel Prices Over Time"):
    """
    Create an interactive line chart for fuel prices over time
    
    Args:
        df: DataFrame with price data
        products: List of product columns to plot
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['#f46530', '#00ff00', '#00bfff', '#ff69b4']
    
    for i, product in enumerate(products):
        if product in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[product],
                mode='lines',
                name=product,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x}<br>' +
                              'Price: Q%{y:.2f}<br>' +
                              '<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (Q)",
        hovermode='x unified',
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_price_distribution(df, products, bins=30):
    """
    Create histogram for price distribution
    
    Args:
        df: DataFrame with price data
        products: List of product columns to plot
        bins: Number of histogram bins
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['#f46530', '#00ff00', '#00bfff', '#ff69b4']
    
    for i, product in enumerate(products):
        if product in df.columns:
            fig.add_trace(go.Histogram(
                x=df[product].dropna(),
                name=product,
                opacity=0.7,
                nbinsx=bins,
                marker_color=colors[i % len(colors)],
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Price Range: %{x}<br>' +
                              'Count: %{y}<br>' +
                              '<extra></extra>'
            ))
    
    fig.update_layout(
        title="Price Distribution",
        xaxis_title="Price (Q)",
        yaxis_title="Frequency",
        barmode='overlay',
        template="plotly_dark",
        height=400
    )
    
    return fig


def create_monthly_avg_price(df, product='Superior'):
    """
    Create bar chart showing average monthly prices
    
    Args:
        df: DataFrame with price data (index should be datetime)
        product: Column name to analyze
    
    Returns:
        Plotly figure object
    """
    if product not in df.columns:
        raise ValueError(f"Column '{product}' not found in DataFrame")
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    df = df.dropna(subset=[product])
    df['Month'] = df.index.month
    
    monthly_mean = df.groupby('Month')[product].mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    best_month = monthly_mean.idxmin()
    
    # Create colors array - highlight best month
    colors = ['#f46530' if i+1 != best_month else '#ff9e7a' 
              for i in range(len(month_names))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=month_names,
        y=monthly_mean.values,
        marker_color=colors,
        text=monthly_mean.values.round(2),
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>' +
                      'Avg Price: Q%{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Average Monthly Prices - {product}",
        xaxis_title="Month",
        yaxis_title="Average Price (Q)",
        template="plotly_dark",
        height=400
    )
    
    return fig


def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numeric columns
    
    Args:
        df: DataFrame with numeric columns
    
    Returns:
        Plotly figure object
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_dark",
        height=600,
        width=800
    )
    
    return fig


def create_consumption_prediction_chart(dates, actual_data, predictions_dict, 
                                        title="Consumption Predictions"):
    """
    Create a chart showing actual consumption and multiple model predictions
    
    Args:
        dates: Array of dates
        actual_data: Array of actual consumption values
        predictions_dict: Dictionary with model names as keys and prediction arrays as values
                         Example: {'LSTM 1': [vals], 'Prophet': [vals]}
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_data,
        mode='lines',
        name='Actual',
        line=dict(color='white', width=3),
        hovertemplate='<b>Actual</b><br>' +
                      'Date: %{x}<br>' +
                      'Consumption: %{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add predictions
    colors = ['#f46530', '#00ff00', '#00bfff', '#ff69b4', '#ffa500']
    dash_styles = ['dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines',
            name=model_name,
            line=dict(
                color=colors[i % len(colors)],
                width=2,
                dash=dash_styles[i % len(dash_styles)]
            ),
            hovertemplate=f'<b>{model_name}</b><br>' +
                          'Date: %{x}<br>' +
                          'Prediction: %{y:,.0f}<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Consumption (barrels)",
        hovermode='x unified',
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_prediction_confidence_interval(dates, actual, prediction, 
                                          lower_bound, upper_bound,
                                          model_name="Model"):
    """
    Create a chart with prediction confidence intervals
    
    Args:
        dates: Array of dates
        actual: Actual values
        prediction: Predicted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        model_name: Name of the model
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates,
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=lower_bound,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(244, 101, 48, 0.3)',
        fill='tonexty',
        name='95% Confidence Interval',
        hovertemplate='Lower: %{y:,.0f}<extra></extra>'
    ))
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='white', width=2),
        hovertemplate='<b>Actual</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=prediction,
        mode='lines',
        name=model_name,
        line=dict(color='#f46530', width=2, dash='dash'),
        hovertemplate=f'<b>{model_name}</b><br>%{{y:,.0f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{model_name} - Prediction with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Consumption (barrels)",
        hovermode='x unified',
        template="plotly_dark",
        height=500
    )
    
    return fig


def create_metrics_comparison_bar(metrics_df, metric_name):
    """
    Create bar chart comparing a specific metric across models
    
    Args:
        metrics_df: DataFrame with columns ['Model', metric_name]
        metric_name: Name of the metric to plot (e.g., 'MAE', 'RMSE')
    
    Returns:
        Plotly figure object
    """
    metric_data = metrics_df[['Model', metric_name]].dropna()
    
    # Find best (minimum) value
    best_idx = metric_data[metric_name].idxmin()
    colors = ['#ff9e7a' if i == best_idx else '#f46530' 
              for i in range(len(metric_data))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metric_data['Model'],
        y=metric_data[metric_name],
        text=metric_data[metric_name].round(2),
        textposition='auto',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>' +
                      f'{metric_name}: %{{y:,.2f}}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{metric_name} Comparison",
        xaxis_title="Model",
        yaxis_title=metric_name,
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig


def create_radar_comparison(metrics_df):
    """
    Create radar chart comparing multiple metrics across models
    
    Args:
        metrics_df: DataFrame with columns ['Model', 'MAE', 'RMSE', 'AIC', 'BIC']
    
    Returns:
        Plotly figure object
    """
    # Normalize metrics (invert so higher is better)
    df_normalized = metrics_df.copy()
    
    metrics_to_normalize = ['MAE', 'RMSE', 'AIC', 'BIC']
    
    for col in metrics_to_normalize:
        if col in df_normalized.columns:
            max_val = df_normalized[col].max()
            min_val = df_normalized[col].min()
            if max_val != min_val and not pd.isna(max_val):
                # Invert: lower values become higher scores
                df_normalized[col] = 1 - ((df_normalized[col] - min_val) / (max_val - min_val))
            else:
                df_normalized[col] = 0.5
    
    fig = go.Figure()
    
    colors = ['#f46530', '#00ff00', '#00bfff', '#ff69b4', '#ffa500']
    
    for idx, row in df_normalized.iterrows():
        values = []
        labels = []
        for metric in metrics_to_normalize:
            if not pd.isna(row[metric]):
                values.append(row[metric])
                labels.append(metric)
        
        if values:
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=row['Model'],
                line=dict(color=colors[idx % len(colors)])
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
        height=600,
        title="Multi-Metric Model Comparison<br><sub>(Higher is better - normalized scores)</sub>"
    )
    
    return fig


def create_year_comparison(df1, df2, year1, year2, product='Superior'):
    """
    Create comparison chart for two years
    
    Args:
        df1: DataFrame for first year
        df2: DataFrame for second year
        year1: Label for first year
        year2: Label for second year
        product: Product column to compare
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if product in df1.columns:
        fig.add_trace(go.Scatter(
            x=df1.index,
            y=df1[product],
            mode='lines',
            name=f'{product} - {year1}',
            line=dict(color='#f46530', width=2)
        ))
    
    if product in df2.columns:
        fig.add_trace(go.Scatter(
            x=df2.index,
            y=df2[product],
            mode='lines',
            name=f'{product} - {year2}',
            line=dict(color='#00ff00', width=2)
        ))
    
    fig.update_layout(
        title=f"{product} Price Comparison: {year1} vs {year2}",
        xaxis_title="Date",
        yaxis_title="Price (Q)",
        hovermode='x unified',
        template="plotly_dark",
        height=500
    )
    
    return fig


def create_error_distribution(actual, predicted, model_name="Model"):
    """
    Create histogram of prediction errors
    
    Args:
        actual: Actual values
        predicted: Predicted values
        model_name: Name of the model
    
    Returns:
        Plotly figure object
    """
    errors = np.array(actual) - np.array(predicted)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        marker_color='#f46530',
        opacity=0.7,
        name='Errors'
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title=f"{model_name} - Prediction Error Distribution",
        xaxis_title="Error (Actual - Predicted)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig