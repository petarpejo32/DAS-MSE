import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TechnicalAnalyzer:
    def __init__(self, stock_data):
        self.df = pd.DataFrame(list(stock_data.values()))
        if not self.df.empty:
            self.df = self.df.sort_values('date')
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

            # Convert all price and volume columns to float
            numeric_columns = ['last_trade_price', 'max_price', 'min_price',
                               'avg_price', 'volume', 'turnover_best', 'total_turnover']
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Fill missing or zero prices with last_trade_price
            price_columns = ['max_price', 'min_price', 'avg_price']
            for col in price_columns:
                self.df.loc[self.df[col].isna() | (self.df[col] == 0), col] = self.df['last_trade_price']

    def get_period_data(self, period_days: int) -> pd.DataFrame:
        """Get data for specific time period."""
        end_date = self.df['date'].max()
        start_date = end_date - pd.Timedelta(days=period_days)
        return self.df[self.df['date'] >= start_date]

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate RSI with proper gain/loss handling"""
        delta = df['last_trade_price'].diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.Series(index=rsi.index, data='hold')
        signals[rsi < 30] = 'buy'
        signals[rsi > 70] = 'sell'

        return rsi.fillna(50), signals

    def calculate_macd(self, df: pd.DataFrame, short: int = 12, long: int = 26, signal: int = 9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """Calculate MACD and generate buy/sell signals"""
        exp1 = df['last_trade_price'].ewm(span=short, adjust=False).mean()
        exp2 = df['last_trade_price'].ewm(span=long, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        signals = pd.Series(index=macd_line.index, data='hold')
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 'buy'
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = 'sell'

        return macd_line, signal_line, signals

    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Stochastic with proper high/low handling"""
        high_period = df['max_price'].rolling(window=period).max()
        low_period = df['min_price'].rolling(window=period).min()
        close = df['last_trade_price']

        # Calculate %K
        k = 100 * ((close - low_period) / (high_period - low_period))
        # Calculate %D (3-period SMA of %K)
        d = k.rolling(window=3).mean()

        signals = pd.Series(index=k.index, data='hold')
        signals[k < 20] = 'buy'
        signals[k > 80] = 'sell'

        return k.fillna(50), d.fillna(50), signals

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Williams %R with proper price range handling"""
        high = df['max_price'].rolling(window=period).max()
        low = df['min_price'].rolling(window=period).min()
        close = df['last_trade_price']

        wr = -100 * ((high - close) / (high - low))

        signals = pd.Series(index=wr.index, data='hold')
        signals[wr < -80] = 'buy'
        signals[wr > -20] = 'sell'

        return wr.fillna(-50), signals

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate CCI with proper typical price calculation"""
        typical_price = (df['max_price'] + df['min_price'] + df['last_trade_price']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        cci = (typical_price - sma_tp) / (0.015 * mad)

        signals = pd.Series(index=cci.index, data='hold')
        signals[cci < -100] = 'buy'
        signals[cci > 100] = 'sell'

        return cci.fillna(0), signals

    def calculate_sma(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate SMA with proper signal generation"""
        sma = df['last_trade_price'].rolling(window=period, min_periods=1).mean()

        signals = pd.Series(index=sma.index, data='hold')
        price = df['last_trade_price']

        # Generate signals for price crossing SMA
        signals[(price > sma) & (price.shift(1) <= sma.shift(1))] = 'buy'
        signals[(price < sma) & (price.shift(1) >= sma.shift(1))] = 'sell'

        return sma.fillna(method='bfill'), signals

    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate EMA and generate buy/sell signals"""
        ema = df['last_trade_price'].ewm(span=period, adjust=False).mean()

        signals = pd.Series(index=ema.index, data='hold')
        price = df['last_trade_price']
        signals[(price > ema) & (price.shift(1) <= ema.shift(1))] = 'buy'
        signals[(price < ema) & (price.shift(1) >= ema.shift(1))] = 'sell'

        return ema, signals

    def calculate_wma(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate WMA with proper weight handling"""
        weights = np.arange(1, period + 1)
        wma = df['last_trade_price'].rolling(window=period, min_periods=1).apply(
            lambda x: np.sum(weights * x) / weights.sum() if len(x) >= period else x.mean()
        )

        signals = pd.Series(index=wma.index, data='hold')
        price = df['last_trade_price']

        signals[(price > wma) & (price.shift(1) <= wma.shift(1))] = 'buy'
        signals[(price < wma) & (price.shift(1) >= wma.shift(1))] = 'sell'

        return wma.fillna(method='bfill'), signals

    def calculate_hull_ma(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Hull MA with proper handling of intermediate calculations"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        # Calculate weighted moving averages
        wma1 = self.calculate_wma(df, half_period)[0]
        wma2 = self.calculate_wma(df, period)[0]

        # Calculate the difference
        diff = 2 * wma1 - wma2

        # Create temporary DataFrame for final WMA calculation
        temp_df = pd.DataFrame({'last_trade_price': diff})
        hma = self.calculate_wma(temp_df, sqrt_period)[0]

        signals = pd.Series(index=hma.index, data='hold')
        price = df['last_trade_price']

        signals[(price > hma) & (price.shift(1) <= hma.shift(1))] = 'buy'
        signals[(price < hma) & (price.shift(1) >= hma.shift(1))] = 'sell'

        return hma.fillna(method='bfill'), signals

    def calculate_dema(self, df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate DEMA and generate buy/sell signals"""
        ema1 = df['last_trade_price'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2

        signals = pd.Series(index=dema.index, data='hold')
        price = df['last_trade_price']
        signals[(price > dema) & (price.shift(1) <= dema.shift(1))] = 'buy'
        signals[(price < dema) & (price.shift(1) >= dema.shift(1))] = 'sell'

        return dema, signals

    def analyze(self, indicators: List[str], duration: int = 30) -> Dict:
        """Analyze selected indicators for the specified time period."""
        if self.df.empty:
            return {'error': 'No data available'}

        # Get data for the specified period
        period_data = self.get_period_data(duration)

        results = {
            'dates': period_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': period_data['last_trade_price'].tolist(),
            'indicators': {}
        }

        # Calculate selected indicators
        indicator_methods = {
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'stochastic': self.calculate_stochastic,
            'williams': self.calculate_williams_r,
            'cci': self.calculate_cci,
            'sma': self.calculate_sma,
            'ema': self.calculate_ema,
            'wma': self.calculate_wma,
            'hma': self.calculate_hull_ma,
            'dema': self.calculate_dema
        }

        for indicator in indicators:
            if indicator not in indicator_methods:
                continue

            if indicator == 'macd':
                macd_line, signal_line, signals = self.calculate_macd(period_data)
                results['indicators']['macd'] = {
                    'macd_line': macd_line.tolist(),
                    'signal_line': signal_line.tolist(),
                    'signals': signals.tolist()
                }
            elif indicator == 'stochastic':
                k, d, signals = self.calculate_stochastic(period_data)
                results['indicators']['stochastic'] = {
                    'k': k.tolist(),
                    'd': d.tolist(),
                    'signals': signals.tolist()
                }
            else:
                values, signals = indicator_methods[indicator](period_data)
                results['indicators'][indicator] = {
                    'values': values.tolist(),
                    'signals': signals.tolist()
                }

        return results