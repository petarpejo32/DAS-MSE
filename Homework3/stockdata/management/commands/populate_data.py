import os
import pandas as pd
from django.utils import timezone
from django.core.management.base import BaseCommand
from stockdata.models import Issuer, StockPrice
from tqdm import tqdm


def clean_data(df):
    # Convert comma-separated numbers
    numeric_columns = ['LastTradePrice', 'Max', 'Min', 'AvgPrice', 'Volume', 'TurnoverBestMKD', 'TotalTurnoverMKD']
    for col in numeric_columns:
        # Replace NaN with 0 and then clean the numbers
        df[col] = df[col].fillna(0)  # Fill NaN with 0
        df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

    # Convert percentage change, replacing NaN with 0
    df['%chg'] = df['%chg'].fillna(0)
    df['%chg'] = df['%chg'].astype(str).str.replace(',', '.').astype(float)

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Final check for any remaining NaN values
    df = df.fillna(0)

    return df


class Command(BaseCommand):
    help = 'Imports stock data from CSV files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before importing',
        )

    def handle(self, *args, **options):
        try:
            # Clear existing data if --clear flag is used
            if options['clear']:
                self.stdout.write('Clearing existing data...')
                StockPrice.objects.all().delete()  # Delete prices first due to foreign key
                Issuer.objects.all().delete()  # Then delete issuers
                self.stdout.write(self.style.SUCCESS('Existing data cleared!'))

            # Get the path to the data directory relative to this script
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(current_dir, 'data')

            self.stdout.write('Populating issuers...')
            # Step 1: Create issuers
            for filename in tqdm(os.listdir(data_dir)):
                if filename.endswith('.csv'):
                    code = filename.replace('.csv', '')
                    Issuer.objects.get_or_create(
                        code=code,
                        defaults={
                            'name': code,
                            'last_updated': timezone.now()
                        }
                    )

            self.stdout.write(self.style.SUCCESS('Issuers populated successfully!'))
            self.stdout.write('Populating stock prices...')

            # Step 2: Import stock prices
            for filename in tqdm(os.listdir(data_dir)):
                if filename.endswith('.csv'):
                    code = filename.replace('.csv', '')
                    issuer = Issuer.objects.get(code=code)

                    try:
                        df = pd.read_csv(os.path.join(data_dir, filename))
                        df = clean_data(df)

                        stock_prices = []
                        for _, row in df.iterrows():
                            try:
                                stock_price = StockPrice(
                                    issuer=issuer,
                                    date=row['Date'],
                                    last_trade_price=float(row['LastTradePrice']) if pd.notna(
                                        row['LastTradePrice']) else 0,
                                    max_price=float(row['Max']) if pd.notna(row['Max']) else 0,
                                    min_price=float(row['Min']) if pd.notna(row['Min']) else 0,
                                    avg_price=float(row['AvgPrice']) if pd.notna(row['AvgPrice']) else 0,
                                    price_change=float(row['%chg']) if pd.notna(row['%chg']) else 0,
                                    volume=int(row['Volume']) if pd.notna(row['Volume']) else 0,
                                    turnover_best=float(row['TurnoverBestMKD']) if pd.notna(
                                        row['TurnoverBestMKD']) else 0,
                                    total_turnover=float(row['TotalTurnoverMKD']) if pd.notna(
                                        row['TotalTurnoverMKD']) else 0
                                )
                                stock_prices.append(stock_price)
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(
                                    f'Skipping row for {issuer.code} due to error: {str(e)}'
                                ))
                                continue

                        if stock_prices:
                            # Bulk create stock prices for this issuer
                            StockPrice.objects.bulk_create(
                                stock_prices,
                                ignore_conflicts=True  # Skip if record already exists
                            )
                            self.stdout.write(self.style.SUCCESS(
                                f'Successfully imported {len(stock_prices)} records for {issuer.code}'
                            ))
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(
                            f'Error processing file {filename}: {str(e)}'
                        ))
                        continue

            self.stdout.write(self.style.SUCCESS('Import completed successfully!'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during import: {str(e)}'))
            raise e
