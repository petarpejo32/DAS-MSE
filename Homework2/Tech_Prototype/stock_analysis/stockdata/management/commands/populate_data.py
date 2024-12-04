from django.core.management.base import BaseCommand
from stockdata.filters.filter import Filter  # Import your Filter class

class Command(BaseCommand):
    help = 'Populate the database with scraped stock data'

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting the data population process...'))

        # Create an instance of Filter and run the data scraping and population
        filter_instance = Filter()
        filter_instance.fill_data()

        self.stdout.write(self.style.SUCCESS(f"Data population completed! Time taken: {filter_instance.get_time_last_scrape()}"))
