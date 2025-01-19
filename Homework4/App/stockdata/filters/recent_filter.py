# stockdata/filters/recent_filter.py

import os
import time
import csv
from datetime import datetime, timedelta
from threading import Thread
import requests
from bs4 import BeautifulSoup
from .error import DataNotFilledError


class RecentFilter:
    def __init__(self, months_back=2):
        self.months_back = months_back
        self.start = None
        self.end = None
        self.all_companies_data = dict()

        # Setup directories
        self.update_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.updates_dir = os.path.join(self.current_dir, 'data_updates')
        self.timestamp_dir = os.path.join(self.updates_dir, self.update_timestamp)

        # Create directories
        os.makedirs(self.updates_dir, exist_ok=True)
        os.makedirs(self.timestamp_dir, exist_ok=True)

        # Fetch companies after directory setup
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
        try:
            cells = row.find_all('td')
            if len(cells) < 9:
                return None

            return [
                cells[0].text.strip(),  # Date
                cells[1].text.strip(),  # LastTradePrice
                cells[2].text.strip() or '',  # Max
                cells[3].text.strip() or '',  # Min
                cells[4].text.strip() or '',  # AvgPrice
                cells[5].text.strip() or '0,00',  # %chg
                cells[6].text.strip() or '0',  # Volume
                cells[7].text.strip() or '0',  # TurnoverBestMKD
                cells[8].text.strip() or '0'  # TotalTurnoverMKD
            ]
        except (IndexError, AttributeError) as e:
            print(f"Error parsing row: {str(e)}")
            return None

    def __get_site_data(self, soup):
        table = soup.find_all('tbody')
        if len(table) == 0:
            return None
        table = table[0]
        table_rows = table.find_all('tr')
        ret_table = []

        for row in table_rows:
            data = self.extract_row_data(row)
            if data:
                ret_table.append(data)

        return ret_table

    def __get_data_for_period(self, company, start_date, end_date):
        base_url = 'https://www.mse.mk/en/stats/symbolhistory/'
        url = f"{base_url}{company}?FromDate={start_date}&ToDate={end_date}"

        max_retries = 3
        retry_delay = 2
        attempts = 0

        while attempts < max_retries:
            try:
                response = requests.post(url, timeout=(60, 120))
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    data = self.__get_site_data(soup)
                    if data:
                        return data
                    return []
            except (requests.exceptions.RequestException, Exception) as e:
                print(f"Attempt {attempts + 1} failed for {company}: {str(e)}")

            attempts += 1
            if attempts < max_retries:
                time.sleep(retry_delay * attempts)

        print(f"Failed to fetch data for {company} after {max_retries} attempts")
        return []

    def __get_data_from_to(self, company):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months_back * 30)

        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')

        company_info = self.__get_data_for_period(company, start_str, end_str)

        if company_info:
            # Write to CSV with index
            filename = os.path.join(self.timestamp_dir, f"{company}.csv")
            rows_with_index = []
            for idx, row in enumerate(company_info):
                rows_with_index.append([idx] + row)

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['', 'Date', 'LastTradePrice', 'Max', 'Min', 'AvgPrice',
                                 '%chg', 'Volume', 'TurnoverBestMKD', 'TotalTurnoverMKD'])
                writer.writerows(rows_with_index)

            print(f"Saved data for {company} to {filename}")
            self.all_companies_data[company] = company_info

    def fill_data(self):
        self.start = time.time()
        threadpool = []

        for cmp in self.companies:
            thread = Thread(target=self.__get_data_from_to, args=(cmp,))
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
