import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import timedelta
import logging


class CustomLSTM(tf.keras.Model):
    def call(self, inputs):
        return self.model(inputs)

    def __init__(self):
        super(CustomLSTM, self).__init__()
        # Even more conservative architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True,
                                 input_shape=(None, 4),
                                 kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.LSTM(16, activation='tanh',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.02))
        ])


class LSTMPredictor:
    def __init__(self, stock_data):
        self.logger = logging.getLogger(__name__)
        self.original_df = pd.DataFrame(list(stock_data.values()))
        if not self.original_df.empty:
            self.original_df = self.original_df.sort_values('date')
            self.original_df['date'] = pd.to_datetime(self.original_df['date']).dt.date
            self.original_df['last_trade_price'] = pd.to_numeric(self.original_df['last_trade_price'], errors='coerce')
            self.original_df = self.original_df.dropna(subset=['last_trade_price'])

        self.df = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    def get_period_data(self, period_days: int) -> pd.DataFrame:
        """Get data for specific time period."""
        end_date = self.original_df['date'].max()
        start_date = end_date - timedelta(days=period_days)
        return self.original_df[self.original_df['date'] >= start_date]

    def add_technical_indicators(self):
        # More conservative technical indicators
        self.df['MA3'] = self.df['last_trade_price'].rolling(window=3).mean()
        self.df['MA5'] = self.df['last_trade_price'].rolling(window=5).mean()
        self.df['volatility'] = self.df['last_trade_price'].pct_change().rolling(window=5).std()
        self.df['trend'] = (self.df['MA3'] - self.df['MA3'].shift(1)) / self.df['MA3'].shift(1)
        self.df = self.df.bfill()

    def prepare_data(self, sequence_length=5):
        if self.df.empty:
            self.logger.error("DataFrame is empty")
            return None, None, None

        if len(self.df) < sequence_length:
            self.logger.error(f"Insufficient data: {len(self.df)} rows, {sequence_length} required")
            return None, None, None

        features = ['last_trade_price', 'MA3', 'MA5', 'volatility']
        feature_data = self.df[features].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)

        x_values, y_values = [], []
        for i in range(sequence_length, len(scaled_features)):
            x_values.append(scaled_features[i - sequence_length:i])
            y_values.append(scaled_features[i, 0])

        return np.array(x_values), np.array(y_values)

    def train_model(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=16):
        model = CustomLSTM()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        loss_fn = tf.keras.losses.Huber(delta=0.5)  # More conservative delta

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        best_val_loss = float('inf')
        patience = 5
        patience_count = 0
        training_loss = 0
        validation_loss = 0

        for epoch in range(epochs):
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    y_pred = model.model(x_batch, training=True)
                    loss = loss_fn(y_batch, y_pred)

                grads = tape.gradient(loss, model.model.trainable_variables)
                # More conservative gradient clipping
                clipped_grads = [tf.clip_by_value(g, -0.5, 0.5) for g in grads]
                optimizer.apply_gradients(zip(clipped_grads, model.model.trainable_variables))
                training_loss = float(loss)

            val_losses = []
            for x_batch, y_batch in val_dataset:
                y_pred = model.model(x_batch, training=False)
                val_loss = loss_fn(y_batch, y_pred)
                val_losses.append(float(val_loss))
            validation_loss = np.mean(val_losses)

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        return model, training_loss, validation_loss

    def calculate_volatility_constraints(self, data, window=30):
        """Calculate very conservative price change constraints based on historical volatility"""
        daily_returns = data['last_trade_price'].pct_change()
        volatility = daily_returns.rolling(window=window).std().mean()

        # Much more conservative limits
        max_up_change = min(volatility, 0.02)  # Max 2% up per day
        max_down_change = max(-volatility, -0.02)  # Max 2% down per day

        return max_up_change, max_down_change

    def predict(self, prediction_horizon):
        try:
            training_days = max(90, prediction_horizon * 30)  # Use more historical data
            self.df = self.get_period_data(training_days)
            if self.df.empty:
                return {'error': 'No data available for the selected period'}

            self.add_technical_indicators()

            # Calculate very conservative volatility constraints
            max_up_change, max_down_change = self.calculate_volatility_constraints(self.df)

            sequence_length = 5
            x_values, y_values = self.prepare_data(sequence_length)

            if x_values is None:
                return {'error': 'Insufficient data for prediction'}

            train_size = int(0.8 * len(x_values))
            x_train = x_values[:train_size]
            x_val = x_values[train_size:]
            y_train = y_values[:train_size]
            y_val = y_values[train_size:]

            model, training_loss, validation_loss = self.train_model(x_train, y_train, x_val, y_val)

            last_sequence = x_values[-1:]
            predictions = []
            current_sequence = last_sequence.copy()
            last_price = float(self.df['last_trade_price'].iloc[-1])

            # Calculate average daily change from last 30 days for trend
            recent_trend = self.df['last_trade_price'].pct_change().tail(30).mean()
            # Limit the trend influence
            recent_trend = max(min(recent_trend, 0.01), -0.01)

            # Calculate moving average for trend comparison
            ma20 = self.df['last_trade_price'].rolling(window=20).mean().iloc[-1]
            price_vs_ma = (last_price - ma20) / ma20

            for i in range(prediction_horizon):
                pred = model.model(current_sequence)
                pred_scaled = pred[0, 0]
                pred_price = self.feature_scaler.inverse_transform(
                    [[pred_scaled, 0, 0, 0]]
                )[0, 0]

                # Calculate percentage change
                pct_change = (pred_price - last_price) / last_price

                # Apply very strong dampening
                damping_factor = 0.3 ** (i + 1)  # Increased from 0.85 to 0.90

                # Consider price position relative to MA
                ma_factor = -price_vs_ma * 0.01 * (0.3 ** i)  # Mean reversion factor

                # Combine all factors with heavy dampening
                trend_factor = recent_trend * (0.7 ** i)  # Reduced trend influence
                final_change = (pct_change * damping_factor + trend_factor + ma_factor) * 0.5

                # Apply strict constraints
                final_change = max(min(final_change, max_up_change), max_down_change)

                # Calculate new price
                new_price = last_price * (1 + final_change)
                predictions.append(new_price)

                # Update for next iteration
                new_scaled = self.feature_scaler.transform([[new_price, 0, 0, 0]])[0]
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_scaled
                last_price = new_price

            # Generate weekday-only future dates
            last_date = self.df['date'].max()
            future_dates = []
            current_date = last_date

            for _ in range(prediction_horizon):
                current_date += timedelta(days=1)
                while current_date.weekday() > 4:
                    current_date += timedelta(days=1)
                future_dates.append(current_date)

            return {
                'historical_dates': [str(date) for date in self.df['date']],
                'historical_prices': self.df['last_trade_price'].tolist(),
                'predicted_prices': predictions,
                'prediction_dates': [str(date) for date in future_dates],
                'training_loss': training_loss,
                'validation_loss': validation_loss
            }

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}
