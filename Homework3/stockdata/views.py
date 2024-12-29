from django.http import JsonResponse
from django.shortcuts import render
from django.template import loader
from .models import Issuer, StockPrice
from .analysis.technical import TechnicalAnalyzer
from .analysis.fundamental import FundamentalAnalyzer
from .analysis.lstm import LSTMPredictor


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
    return render(request, 'predict.html')


def analyze(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    analysis_type = request.POST.get('analysis_type')
    company = request.POST.get('company')
    duration = int(request.POST.get('duration', 30))

    try:
        # Get issuer from the database
        issuer = Issuer.objects.get(code=company)

        # Fetch the stock data for the selected issuer and time range
        stock_data = StockPrice.objects.filter(issuer=issuer).order_by('-date')[:duration]

        # If the user selects technical analysis
        if analysis_type == 'technical':
            analyzer = TechnicalAnalyzer(stock_data)
            selected_indicators = request.POST.getlist('indicators')
            results = analyzer.analyze(selected_indicators)

        # If the user selects fundamental analysis
        elif analysis_type == 'fundamental':
            analyzer = FundamentalAnalyzer(issuer)
            news_period = int(request.POST.get('news_period', 30))
            results = analyzer.analyze(news_period)

        # If the user selects LSTM-based prediction
        elif analysis_type == 'lstm':
            predictor = LSTMPredictor(stock_data)
            prediction_horizon = int(request.POST.get('prediction_horizon', 7))
            results = predictor.predict(prediction_horizon)

        else:
            return JsonResponse({'error': 'Invalid analysis type'}, status=400)

        # Return the results of the analysis as a JSON response
        return JsonResponse(results)

    except Issuer.DoesNotExist:
        return JsonResponse({'error': 'Company not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


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
