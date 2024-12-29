import math

import pandas as pd


class TechnicalAnalyzer:
    def __init__(self, stock_data):
        # Convert the queryset to a pandas DataFrame
        self.df = pd.DataFrame(list(stock_data.values()))
        if not self.df.empty:
            # Sort the DataFrame by the date
            self.df = self.df.sort_values('date')

            # Convert the 'date' column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

            # Convert 'last_trade_price' to float
            self.df['last_trade_price'] = self.df['last_trade_price'].astype(float)

    def calculate_rsi(self, period=14):
        delta = self.df['last_trade_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        result = 100 - (100 / (1 + rs))
        print(result)
        return result

    def calculate_macd(self, short=12, long=26, signal=9):
        exp1 = self.df['last_trade_price'].ewm(span=short, adjust=False).mean()
        exp2 = self.df['last_trade_price'].ewm(span=long, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def calculate_sma(self, period):
        return self.df['last_trade_price'].rolling(window=period).mean()

    def calculate_ema(self, period):
        return self.df['last_trade_price'].ewm(span=period, adjust=False).mean()

    def calculate_williams_r(self, period=14):
        high = self.df['max_price'].rolling(window=period).max()
        low = self.df['min_price'].rolling(window=period).min()
        close = self.df['last_trade_price']
        wr = ((high - close) / (high - low)) * -100
        return wr

    def calculate_stochastic(self, period=14):
        high = self.df['max_price'].rolling(window=period).max()
        low = self.df['min_price'].rolling(window=period).min()
        close = self.df['last_trade_price']
        k = ((close - low) / (high - low)) * 100
        d = k.rolling(window=3).mean()
        print(k, d)
        return k, d

    def analyze(self, indicators):
        # indactors = ['macd', 'dema']
        if self.df.empty:
            return {'error': 'No data available'}

        results = {
            'dates': self.df['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': self.df['last_trade_price'].tolist(),
            'indicators': {}
        }

        for indicator in indicators:
            if indicator == 'rsi':
                results['indicators']['rsi'] = self.calculate_rsi().tolist()
            elif indicator == 'macd':
                macd_line, signal = self.calculate_macd()
                results['indicators']['macd'] = {
                    'macd_line': macd_line.tolist(),
                    'signal': signal.tolist()
                }
            elif indicator == 'sma':
                results['indicators']['sma'] = self.calculate_sma(20).tolist()
            elif indicator == 'ema':
                results['indicators']['ema'] = self.calculate_ema(20).tolist()
            elif indicator == 'williams':
                results['indicators']['williams'] = self.calculate_williams_r().tolist()
            elif indicator == 'stochastic':
                k, d = self.calculate_stochastic()
                results['indicators']['stochastic'] = {
                    'k': k.tolist(),
                    'd': d.tolist()
                }

        return results
