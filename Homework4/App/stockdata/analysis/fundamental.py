import pandas as pd
from datetime import datetime, timedelta


class FundamentalAnalyzer:
    def __init__(self, issuer):
        self.issuer = issuer

    def analyze(self, news_period):
        try:
            print(f"Starting analysis for {self.issuer.code}")

            # Get the latest record and print its date
            latest_record = self.issuer.stockprice_set.order_by('-date').first()
            print(f"Latest record date: {latest_record.date}")

            if not latest_record:
                return {
                    'error': 'No data available for this company',
                    'dates': [],
                    'prices': [],
                    'sentiment_trend': [],
                    'recommendation': 'neutral'
                }

            # Get all records up to the latest date, limited by news_period
            stock_data = self.issuer.stockprice_set.order_by('-date')[:news_period]
            print(f"Found {stock_data.count()} records")

            # Process the data
            sentiments = []
            stock_list = list(stock_data)  # Convert to list to process in reverse
            stock_list.reverse()  # Process from oldest to newest

            previous_price = None
            for stock in stock_list:
                current_price = float(stock.last_trade_price)
                print(f"Processing record date: {stock.date}, price: {current_price}")

                if previous_price is not None:
                    price_change = ((current_price - previous_price) / previous_price) * 100

                    sentiments.append({
                        'date': stock.date.strftime('%Y-%m-%d'),
                        'sentiment': price_change / 100,
                        'price': current_price
                    })

                previous_price = current_price

            if not sentiments:
                return {
                    'error': 'Insufficient data for analysis',
                    'dates': [],
                    'prices': [],
                    'sentiment_trend': [],
                    'recommendation': 'neutral'
                }

            df = pd.DataFrame(sentiments)
            print(f"Created DataFrame with {len(df)} rows")

            results = {
                'dates': df['date'].tolist(),
                'prices': df['price'].tolist(),
                'sentiment_trend': df['sentiment'].tolist(),
                'average_sentiment': float(df['sentiment'].mean()),
                'recommendation': 'buy' if df['sentiment'].mean() > 0 else 'sell'
            }

            return results

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {
                'error': f'Analysis error: {str(e)}',
                'dates': [],
                'prices': [],
                'sentiment_trend': [],
                'recommendation': 'neutral'
            }