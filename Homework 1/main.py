from filter import Filter

if __name__ == '__main__':
    data_filter = Filter()
    data_filter.fill_data()
    print(len(data_filter.all_companies_data))
    print(data_filter.get_time_last_scrape())
