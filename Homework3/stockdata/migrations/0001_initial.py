# Generated by Django 5.1.3 on 2024-12-03 17:40

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Issuer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('code', models.CharField(max_length=20, unique=True)),
                ('name', models.CharField(max_length=200)),
                ('last_updated', models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='StockPrice',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('last_trade_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('max_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('min_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('avg_price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('price_change', models.DecimalField(decimal_places=2, max_digits=10)),
                ('volume', models.IntegerField()),
                ('turnover_best', models.DecimalField(decimal_places=2, max_digits=15)),
                ('total_turnover', models.DecimalField(decimal_places=2, max_digits=15)),
                ('issuer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='stockdata.issuer')),
            ],
            options={
                'indexes': [models.Index(fields=['date'], name='stockdata_s_date_d6a050_idx'), models.Index(fields=['issuer', 'date'], name='stockdata_s_issuer__19f47a_idx')],
                'unique_together': {('issuer', 'date')},
            },
        ),
    ]