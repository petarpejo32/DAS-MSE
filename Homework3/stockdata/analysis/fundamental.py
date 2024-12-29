import pandas as pd
from datetime import datetime, timedelta


class FundamentalAnalyzer:
    def __init__(self, issuer):
        self.issuer = issuer

    def analyze(self, news_period):
        try:
            # Get actual stock data from the database
            end_date = datetime.now()
            start_date = end_date - timedelta(days=news_period)

            stock_data = self.issuer.stockprice_set.filter(
                date__range=(start_date, end_date)
            ).order_by('date')

            if not stock_data:
                return {
                    'error': 'No data available for selected period',
                    'average_sentiment': 0,
                    'sentiment_trend': [],
                    'dates': [],
                    'recommendation': 'neutral'
                }

            # Use price changes as sentiment indicators
            sentiments = []
            for stock in stock_data:
                sentiments.append({
                    'date': stock.date.strftime('%Y-%m-%d'),
                    'sentiment': float(stock.price_change) / 100,  # Convert to -1 to 1 scale
                    'price': float(stock.last_trade_price)
                })

            df = pd.DataFrame(sentiments)

            return {
                'average_sentiment': float(df['sentiment'].mean()),
                'sentiment_trend': df['sentiment'].tolist(),
                'dates': df['date'].tolist(),
                'prices': df['price'].tolist(),
                'recommendation': 'buy' if df['sentiment'].mean() > 0 else 'sell'
            }
        except Exception as e:
            return {
                'error': str(e),
                'average_sentiment': 0,
                'sentiment_trend': [],
                'dates': [],
                'prices': [],
                'recommendation': 'neutral'
            }
