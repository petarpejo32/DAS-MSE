# Generated by Django 5.1.3 on 2025-01-19 21:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockdata', '0002_alter_issuer_options_alter_stockprice_options_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='issuer',
            name='last_updated',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
