import time
from datetime import datetime, timedelta
from decimal import Decimal
from threading import Thread

import requests
from bs4 import BeautifulSoup

from .error import DataNotFilledError
from ..models import Issuer, StockPrice


class Filter:
    start = None
    end = None
    all_companies_data = dict()

    def __init__(self):
        self.companies = self.__fetch_companies()

    def __fetch_companies(self):
        sample = 'https://www.mse.mk/en/stats/symbolhistory/TEL'

        response = requests.get(sample)
        soup = BeautifulSoup(response.text, 'html.parser')

        company_names = soup.select('.form-control option')

        valid_company_names = []

        for name in company_names:
            if not (any(char.isdigit() for char in name.text)):
                valid_company_names.append(name.text)

        return valid_company_names

    def extract_row_data(self, row):
        translation_table = str.maketrans({',': '.', '.': ','})

        cells = row.find_all('td')
        original_date = cells[0].text
        date = datetime.strptime(original_date, '%m/%d/%Y').strftime('%d/%m/%Y')

        last_trade_price = Decimal(cells[1].text.translate(translation_table))
        max_price = Decimal(cells[2].text.translate(translation_table))
        min_price = Decimal(cells[3].text.translate(translation_table))
        avg_price = Decimal(cells[4].text.translate(translation_table))
        percent_change = Decimal(cells[5].text.translate(translation_table))
        volume = int(cells[6].text.replace(',', ''))
        turnover_best_mkd = Decimal(cells[7].text.translate(translation_table))
        total_turnover_mkd = Decimal(cells[8].text.translate(translation_table))

        return [
            date, last_trade_price, max_price, min_price, avg_price,
            percent_change, volume, turnover_best_mkd, total_turnover_mkd
        ]

    def __get_site_data(self, soup):
        table = soup.find_all('tbody')
        if len(table) == 0:
            return None
        table = table[0]
        table_rows = table.find_all('tr')
        ret_table = []

        for row in table_rows:
            ret_table.append(self.extract_row_data(row))

        return ret_table

    def __get_x_days_ago_of(self, date, days):
        date_from = datetime.strptime(date, '%m/%d/%Y') - timedelta(days=days)
        return date_from.strftime('%m/%d/%Y')

    def __get_data_from_to(self, company, date_from, date_to):
        base_url = 'https://www.mse.mk/en/stats/symbolhistory/'

        date_from_obj = datetime.strptime(date_from, '%m/%d/%Y')
        date_to_obj = datetime.strptime(date_to, '%m/%d/%Y')

        days = (date_to_obj - date_from_obj).days
        years = days // 365
        daysleft = days % 365

        date_from = self.__get_x_days_ago_of(date_to, 365)

        if years == 0:
            date_from = self.__get_x_days_ago_of(date_to, daysleft)

        company_info = []
        attempts = 0
        max_attempts = 5
        response = None  # Initialize response variable

        for i in range(1, years + 2):
            if i == (years + 1):
                date_from = datetime.strptime(date_to, '%m/%d/%Y') - timedelta(days=daysleft)
                date_from = date_from.strftime('%m/%d/%Y')

            url = base_url + company + "?" + "FromDate=" + date_from + '&ToDate=' + date_to

            while attempts < max_attempts:
                try:
                    response = requests.post(url, timeout=(60, 120))
                    if response.status_code == 200:
                        break
                    else:
                        attempts += 1
                        print(f"Failed to fetch data for {company}. Status code: {response.status_code}")
                        if attempts < max_attempts:
                            time.sleep(2 ** attempts)  # Exponential backoff
                        else:
                            raise DataNotFilledError(f"Failed to fetch data for {company} after {max_attempts} attempts.")
                except requests.exceptions.Timeout:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise DataNotFilledError(f"Timeout error fetching data for {company}.")
                    time.sleep(2 ** attempts)  # Exponential backoff
                except requests.exceptions.RequestException as e:
                    raise DataNotFilledError(f"Error fetching data for {company}: {e}")

            # Check if response is None after retries
            if response is None:
                raise DataNotFilledError(f"Failed to get a valid response for {company} after {max_attempts} attempts.")

            soup = BeautifulSoup(response.text, 'html.parser')
            data_append = self.__get_site_data(soup)

            if data_append is not None:
                company_info += data_append

            date_to = date_from
            date_from = self.__get_x_days_ago_of(date_to, 365)

        # Now save to the database instead of CSV
        for row in company_info:
            date = datetime.strptime(row[0], '%d/%m/%Y')
            last_trade_price = row[1]
            max_price = row[2]
            min_price = row[3]
            avg_price = row[4]
            price_change = row[5]
            volume = row[6]
            turnover_best_mkd = row[7]
            total_turnover = row[8]

            # Get or create Issuer
            issuer, created = Issuer.objects.get_or_create(
                code=company,  # Assuming 'company' is the code
                defaults={'name': company}  # Optionally, you can set the name if needed
            )

            # Create a new StockPrice entry
            StockPrice.objects.create(
                issuer=issuer,
                date=date,
                last_trade_price=last_trade_price,
                max_price=max_price,
                min_price=min_price,
                avg_price=avg_price,
                price_change=price_change,
                volume=volume,
                turnover_best=turnover_best_mkd,
                total_turnover=total_turnover
            )

        self.all_companies_data[company] = company_info

    def fill_data(self):
        dataframes = dict()
        threadpool = []
        today = str(datetime.today().strftime('%m/%d/%Y'))

        self.start = time.time()

        for cmp in self.companies:

            thread = Thread(target=self.__get_data_from_to, args=(cmp, today, today))
            thread.start()
            threadpool.append(thread)

        for thread in threadpool:
            thread.join()

        self.end = time.time()

    def get_time_last_scrape(self):

        if self.end is None:
            raise DataNotFilledError('Please fill the database first by running the fill_data() method')

        time_s = (self.end - self.start)

        return f'{int(time_s / 60)}m {int(time_s % 60)}s'
