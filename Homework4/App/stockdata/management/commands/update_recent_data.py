# stockdata/management/commands/update_recent_data.py

import os
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from stockdata.filters.recent_filter import RecentFilter
from stockdata.models import Issuer, StockPrice
from tqdm import tqdm


def clean_data_recent(df):
    """Clean data function specifically for recent updates"""
    try:
        # Convert comma-separated numbers and handle dots in thousands
        numeric_columns = ['LastTradePrice', 'Max', 'Min', 'AvgPrice', 'Volume', 'TurnoverBestMKD', 'TotalTurnoverMKD']
        for col in numeric_columns:
            # First replace NaN with 0
            df[col] = df[col].fillna('0')
            # Convert to string and clean
            df[col] = df[col].astype(str).apply(lambda x: x.replace('.', '').replace(',', '.'))
            # Convert to float
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert percentage change
        df['%chg'] = df['%chg'].fillna('0')
        df['%chg'] = df['%chg'].astype(str).str.replace(',', '.').astype(float)

        # Convert date from M/D/Y to YY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        return df
    except Exception as e:
        print(f"Error in clean_data: {str(e)}")
        return None


class Command(BaseCommand):
    help = 'Updates stock data for recent months'

    def add_arguments(self, parser):
        parser.add_argument(
            '--months',
            type=int,
            default=2,
            help='Number of months of data to fetch'
        )

    def handle(self, *args, **options):
        months = options['months']
        self.stdout.write(f'Fetching last {months} months of stock data...')

        filter = RecentFilter(months_back=months)
        try:
            filter.fill_data()
            duration = filter.get_time_last_scrape()
            self.stdout.write(self.style.SUCCESS(f'Data scraped to CSV files in {duration}'))

            self.stdout.write('Importing new data...')

            for filename in tqdm(os.listdir(filter.timestamp_dir)):
                if filename.endswith('.csv'):
                    code = filename.replace('.csv', '')
                    try:
                        # Get or create issuer
                        issuer, _ = Issuer.objects.get_or_create(
                            code=code,
                            defaults={'name': code}
                        )

                        # Read and clean the CSV
                        df = pd.read_csv(os.path.join(filter.timestamp_dir, filename))
                        df = clean_data_recent(df)

                        if df is None:
                            self.stdout.write(self.style.WARNING(
                                f'Skipping {code} due to data cleaning error'
                            ))
                            continue

                        # Create stock prices
                        stock_prices = []
                        for _, row in df.iterrows():
                            # Convert date string to date object
                            date_obj = datetime.strptime(row['Date'], '%Y-%m-%d').date()

                            # Check if this record already exists
                            if not StockPrice.objects.filter(
                                    issuer=issuer,
                                    date=date_obj
                            ).exists():
                                try:
                                    stock_price = StockPrice(
                                        issuer=issuer,
                                        date=date_obj,
                                        last_trade_price=row['LastTradePrice'],
                                        max_price=row['Max'],
                                        min_price=row['Min'],
                                        avg_price=row['AvgPrice'],
                                        price_change=row['%chg'],
                                        volume=int(row['Volume']),
                                        turnover_best=row['TurnoverBestMKD'],
                                        total_turnover=row['TotalTurnoverMKD']
                                    )
                                    stock_prices.append(stock_price)
                                except Exception as e:
                                    self.stdout.write(self.style.WARNING(
                                        f'Error processing row for {code}: {str(e)}'
                                    ))
                                    continue

                        if stock_prices:
                            StockPrice.objects.bulk_create(stock_prices, ignore_conflicts=True)
                            self.stdout.write(self.style.SUCCESS(
                                f'Successfully imported {len(stock_prices)} new records for {code}'
                            ))
                        else:
                            self.stdout.write(self.style.SUCCESS(
                                f'No new records to import for {code}'
                            ))

                    except Exception as e:
                        self.stdout.write(self.style.ERROR(
                            f'Error processing file {filename}: {str(e)}'
                        ))
                        continue

            self.stdout.write(self.style.SUCCESS('Update completed successfully!'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during update: {str(e)}'))
