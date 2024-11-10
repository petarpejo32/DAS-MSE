import os
from threading import Thread
import pandas as pd
from bs4 import BeautifulSoup
import time
import requests
from datetime import datetime, timedelta

from error import DataNotFilledError


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

    def __create_table(self, company_data: list):
        df = pd.DataFrame(company_data)
        df.columns = ['Date', 'LastTradePrice', 'Max', 'Min', 'AvgPrice', '%chg', "Volume", 'TurnoverBestMKD', 'TotalTurnoverMKD']
        return df

    def extract_row_data(self, row):
        translation_table = str.maketrans({',': '.', '.': ','})

        cells = row.find_all('td')
        original_date = cells[0].text
        date = datetime.strptime(original_date, '%m/%d/%Y').strftime('%d/%m/%Y')

        last_trade_price = cells[1].text.translate(translation_table)
        max_price = cells[2].text.translate(translation_table)
        min_price = cells[3].text.translate(translation_table)
        avg_price = cells[4].text.translate(translation_table)
        percent_change = cells[5].text.translate(translation_table)
        volume = cells[6].text
        turnover_best_mkd = cells[7].text.translate(translation_table)
        total_turnover_mkd = cells[8].text.translate(translation_table)

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
        return date_from.strftime('%m/%d/%Y').__str__()

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

        for i in range(1, years + 2):

            if i == (years + 1):
                date_from = datetime.strptime(date_to, '%m/%d/%Y') - timedelta(days=daysleft)
                date_from = date_from.strftime('%m/%d/%Y').__str__()

            url = base_url + company + "?" + "FromDate=" + date_from + '&ToDate=' + date_to
            response = requests.post(url, timeout=(25, 60))

            while response.status_code != 200:
                response = requests.post(url, timeout=(25, 60))

            soup = BeautifulSoup(response.text, 'html.parser')

            data_append = self.__get_site_data(soup)

            if data_append is not None:
                company_info += data_append

            date_to = date_from
            date_from = self.__get_x_days_ago_of(date_to, 365)

        table = self.__create_table(company_info)
        table.to_csv(f'./data/{company}.csv')
        self.all_companies_data[company] = table


    def fill_data(self):
        dataframes = dict()
        threadpool = []
        today = str(datetime.today().strftime('%m/%d/%Y'))

        if not os.path.exists('./data'):
            os.makedirs('./data')

        self.start = time.time()

        for cmp in self.companies:

            if f'{cmp}.csv' not in os.listdir('./data'):
                search_from = self.__get_x_days_ago_of(today, 365 * 10)
            else:
                curr_df = pd.read_csv(f'./data/{cmp}.csv')
                search_from = str(
                    (datetime.strptime(curr_df.Date[0], '%m/%d/%Y') + timedelta(days=1)).strftime('%m/%d/%Y'))
                yesterday = (datetime.today() - timedelta(days=1)).strftime('%m/%d/%Y')

                if today == search_from or yesterday == search_from:
                    continue

                dataframes[cmp] = curr_df.drop(columns=['Unnamed: 0'])

            thread = Thread(target=self.__get_data_from_to, args=(cmp, search_from, today))
            thread.start()
            threadpool.append(thread)

        for thread in threadpool:
            thread.join()

        for cmp in self.companies:
            if cmp in self.all_companies_data:
                df = pd.concat(
                    [self.all_companies_data[cmp], dataframes[cmp] if cmp in dataframes.keys() else pd.DataFrame()],
                    axis=0)
                df = df.reset_index().drop(columns=['index'])
                df.to_csv(f'./data/{cmp}.csv')

        self.end = time.time()

    def get_time_last_scrape(self):

        if self.end is None:
            raise DataNotFilledError('Please fill the database first by running the fill_data() method')

        time_s = (self.end - self.start)

        return f'{int(time_s / 60)}m {int(time_s % 60)}s'
