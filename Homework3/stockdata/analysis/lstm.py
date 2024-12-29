# stockdata/analysis/lstm.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


class LSTMPredictor:
    def __init__(self, stock_data):
        # Convert queryset to DataFrame
        self.df = pd.DataFrame(list(stock_data.values()))
        if not self.df.empty:
            self.df = self.df.sort_values('date')
            self.df['last_trade_price'] = self.df['last_trade_price'].astype(float)
        self.scaler = MinMaxScaler()

    def prepare_data(self, sequence_length=60):
        if self.df.empty:
            return None, None, None

        data = self.df['last_trade_price'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        x_values, y_values = [], []
        for i in range(sequence_length, len(scaled_data)):
            x_values.append(scaled_data[i - sequence_length:i, 0])
            y_values.append(scaled_data[i, 0])

        return np.array(x_values), np.array(y_values)

    @staticmethod
    def create_model(sequence_length):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def predict(self, prediction_horizon):
        if self.df.empty:
            return {'error': 'No data available'}

        try:
            sequence_length = 60
            x_values, y_values = self.prepare_data(sequence_length)

            if x_values is None:
                return {'error': 'Insufficient data for prediction'}

            # Split data
            split = int(0.7 * len(x_values))
            x_train = x_values[:split]
            x_test = x_values[split:]
            y_train = y_values[:split]

            # Reshape for LSTM
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

            # Create and train model
            model = self.create_model(sequence_length)
            model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

            # Prepare data for future prediction
            last_sequence = x_test[-1:].reshape((1, sequence_length, 1))

            # Make predictions
            predictions = []
            current_sequence = last_sequence.copy()

            for _ in range(prediction_horizon):
                pred = model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[0, -1, 0] = pred[0, 0]

            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)

            return {
                'historical_dates': self.df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'historical_prices': self.df['last_trade_price'].tolist(),
                'predicted_prices': predictions.flatten().tolist(),
                'prediction_dates': pd.date_range(
                    start=self.df['date'].max(),
                    periods=prediction_horizon + 1
                )[1:].strftime('%Y-%m-%d').tolist()
            }
        except Exception as e:
            return {'error': str(e)}
