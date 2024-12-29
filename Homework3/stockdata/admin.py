from django.contrib import admin
from .models import Issuer, StockPrice

admin.site.register(Issuer)
admin.site.register(StockPrice)
