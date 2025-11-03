import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import joblib

class LSTMPredictor:
    """Class to handle LSTM model loading and predictions with differencing"""
    
    def __init__(self, models_dir='./models'):
        self.models_dir = models_dir
        self.lstm_model_1 = None
        self.lstm_model_2 = None
        self.scaler = None
        self.metadata_1 = None
        self.metadata_2 = None
        self.general_metadata = None
        self.combined_data = None
        
    def load_models(self):
        """Load all trained models and data"""
        try:
            # Load LSTM models with custom objects to handle metric issues
            # Compile=False avoids the metric deserialization issue
            self.lstm_model_1 = keras.models.load_model(
                f'{self.models_dir}/lstm_model_1.h5',
                compile=False
            )
            self.lstm_model_2 = keras.models.load_model(
                f'{self.models_dir}/lstm_model_2.h5',
                compile=False
            )
            
            # Recompile models for prediction (not training)
            # We don't need metrics for prediction, just the model architecture
            self.lstm_model_1.compile(optimizer='adam', loss='mse')
            self.lstm_model_2.compile(optimizer='adam', loss='mse')
            
            # Load scaler
            self.scaler = joblib.load(f'{self.models_dir}/lstm_scaler.pkl')
            
            # Load metadata
            with open(f'{self.models_dir}/lstm_model_1_metadata.pkl', 'rb') as f:
                self.metadata_1 = pickle.load(f)
            
            with open(f'{self.models_dir}/lstm_model_2_metadata.pkl', 'rb') as f:
                self.metadata_2 = pickle.load(f)
            
            with open(f'{self.models_dir}/general_metadata.pkl', 'rb') as f:
                self.general_metadata = pickle.load(f)
            
            # Load combined data
            self.combined_data = pd.read_csv(
                f'{self.models_dir}/combined_data.csv',
                index_col=0,
                parse_dates=True
            )
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_next_n_months(self, model, n_months=30, batch_size=1):
        """
        Generate predictions for the next n months using your prediction logic.
        This follows your prediccion_fun approach with differencing.
        
        Args:
            model: Trained Keras LSTM model
            n_months: Number of months to predict
            batch_size: Batch size for prediction
        
        Returns:
            pd.Series with predictions indexed by dates
        """
        if self.combined_data is None or self.scaler is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Get the last value and prepare for prediction
        series_values = self.combined_data['super'].values
        predictions = []
        current_series = series_values.copy()
        
        # Start predicting from the last known value
        for i in range(n_months):
            # Get the last value (for differencing)
            last_value = current_series[-1]
            
            # Calculate the difference (if needed, but here we predict the difference directly)
            # Create input for model (we need the last difference)
            if len(current_series) >= 2:
                last_diff = current_series[-1] - current_series[-2]
            else:
                last_diff = 0
            
            # Scale the difference
            last_diff_scaled = self.scaler.transform([[last_diff]])[0][0]
            
            # Reshape for LSTM input: (batch_size, timesteps, features)
            X = np.array([[[last_diff_scaled]]])
            
            # Predict the next difference
            yhat_scaled = model.predict(X, batch_size=batch_size, verbose=0)
            
            # Inverse scale
            yhat_diff = self.scaler.inverse_transform(yhat_scaled)[0][0]
            
            # Apply differencing inversion: next_value = current_value + predicted_difference
            next_value = last_value + yhat_diff
            
            # Store prediction
            predictions.append(next_value)
            
            # Update series with new prediction
            current_series = np.append(current_series, next_value)
        
        # Create date range for predictions
        last_date = self.combined_data.index[-1]
        
        # Use pd.date_range with proper freq parameter
        pred_dates = pd.date_range(
            start=last_date,
            periods=n_months + 1,  # +1 because we want to start AFTER last_date
            freq='MS'  # Month start
        )[1:]  # Skip the first date (which is last_date itself)
        
        return pd.Series(predictions, index=pred_dates)
    
    def get_all_predictions(self, n_months=30):
        """Get predictions from both LSTM models"""
        predictions = {}
        
        if self.lstm_model_1 is not None:
            predictions['LSTM Model 1'] = self.predict_next_n_months(
                self.lstm_model_1, 
                n_months=n_months,
                batch_size=self.metadata_1.get('batch_size', 1)
            )
        
        if self.lstm_model_2 is not None:
            predictions['LSTM Model 2 (Tuned)'] = self.predict_next_n_months(
                self.lstm_model_2, 
                n_months=n_months,
                batch_size=self.metadata_2.get('batch_size', 1)
            )
        
        return predictions
    
    def get_historical_data(self):
        """Get historical data for plotting"""
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_models() first.")
        
        return self.combined_data
    
    def get_metrics(self):
        """Get model performance metrics"""
        metrics = {}
        
        if self.metadata_1:
            metrics['LSTM Model 1'] = self.metadata_1['metrics']
        
        if self.metadata_2:
            metrics['LSTM Model 2 (Tuned)'] = self.metadata_2['metrics']
        
        return metrics
    
    def calculate_statistics(self, predictions_dict):
        """Calculate statistics from predictions"""
        stats = {}
        
        for model_name, predictions in predictions_dict.items():
            if len(predictions) > 0:
                # Next month prediction (first value)
                next_month_pred = predictions.iloc[0]
                
                # Average of next 4 months
                next_4_months_avg = predictions.iloc[:4].mean() if len(predictions) >= 4 else predictions.mean()
                
                # Monthly trend (compare first 4 months to last 4 months of predictions)
                if len(predictions) >= 8:
                    first_period = predictions.iloc[:4].mean()
                    last_period = predictions.iloc[-4:].mean()
                    trend = "↗️ Creciente" if last_period > first_period else "↘️ Decreciente"
                    trend_pct = ((last_period - first_period) / first_period) * 100
                else:
                    trend = "→ Estable"
                    trend_pct = 0
                
                stats[model_name] = {
                    'next_month': next_month_pred,
                    'next_4_months_avg': next_4_months_avg,
                    'trend': trend,
                    'trend_pct': trend_pct
                }
        
        # Calculate average across both models
        if stats:
            avg_next_month = np.mean([s['next_month'] for s in stats.values()])
            avg_next_4_months = np.mean([s['next_4_months_avg'] for s in stats.values()])
            
            # Determine overall trend
            increasing = sum(1 for s in stats.values() if "Creciente" in s['trend'])
            decreasing = sum(1 for s in stats.values() if "Decreciente" in s['trend'])
            
            if increasing > decreasing:
                avg_trend = "↗️ Creciente"
            elif decreasing > increasing:
                avg_trend = "↘️ Decreciente"
            else:
                avg_trend = "→ Estable"
            
            avg_trend_pct = np.mean([s['trend_pct'] for s in stats.values()])
            
            stats['Average'] = {
                'next_month': avg_next_month,
                'next_4_months_avg': avg_next_4_months,
                'trend': avg_trend,
                'trend_pct': avg_trend_pct
            }
        
        return stats


def load_and_predict(models_dir='./models', prediction_months=30):
    """
    Convenience function to load models and get predictions.
    
    Args:
        models_dir: Directory where models are stored
        prediction_months: Number of months to predict into the future
    
    Returns:
        tuple: (predictor, predictions, statistics, metrics)
    """
    predictor = LSTMPredictor(models_dir)
    
    if not predictor.load_models():
        return None, None, None, None
    
    predictions = predictor.get_all_predictions(prediction_months)
    statistics = predictor.calculate_statistics(predictions)
    metrics = predictor.get_metrics()
    
    return predictor, predictions, statistics, metrics