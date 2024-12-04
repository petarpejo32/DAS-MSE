from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from stockdata.models import Issuer, StockPrice


# Create your views here.
def index(request):
    template = loader.get_template("index.html")
    context = {}
    return render(request, "index.html", {})

def stocks(request):
    data = Issuer.objects.all()
    return render(request, "list_stocks.html", {
        "stocks": data
    })
def predict(request):
    template = loader.get_template("predict.html")
    return render(request, 'predict.html')

def detailed_stocks(request, stock_id: int):
    issuer = Issuer.objects.get(pk=stock_id)
    stocks = issuer.stockprice_set.all()
    average_price = []
    volumes = []
    prices = []
    for stock in stocks:
        average_price.append(str(stock.avg_price))
        volumes.append(stock.volume)
        prices.append(str(stock.price_change))



    latest_stock = issuer.stockprice_set.latest()
    return render(request, "details_stocks.html", {
        "latest_stock": latest_stock,
        "selected": stocks,
        "issuer": issuer,
        "prices": prices,
        "average_prices": average_price,
        "volumes": volumes
    })