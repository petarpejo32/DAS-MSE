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

    try:
        analysis_type = request.POST.get('analysis_type')
        company = request.POST.get('company')
        duration = int(request.POST.get('duration', 30))

        if not company:
            return JsonResponse({'error': 'Please enter a company code'}, status=400)

        # Get issuer from the database
        issuer = Issuer.objects.get(code=company.upper())

        # Fetch the stock data for the selected issuer
        stock_data = StockPrice.objects.filter(issuer=issuer).order_by('-date')

        # Check if we have enough data
        if not stock_data.exists():
            return JsonResponse({'error': 'No data available for this company'}, status=404)

        # If the user selects technical analysis
        if analysis_type == 'technical':
            selected_indicators = request.POST.getlist('indicators')
            if not selected_indicators:
                return JsonResponse({'error': 'Please select at least one indicator'}, status=400)

            analyzer = TechnicalAnalyzer(stock_data)
            results = analyzer.analyze(selected_indicators, duration)

        # If the user selects fundamental analysis

        elif analysis_type == 'fundamental':

            analyzer = FundamentalAnalyzer(issuer)
            news_period = int(request.POST.get('news_period', duration))
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
        return JsonResponse({'error': 'Company not found. Please enter a valid company code.'}, status=404)
    except ValueError as e:
        return JsonResponse({'error': f'Invalid value: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Analysis error: {str(e)}'}, status=500)


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