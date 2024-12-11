import pandas as pd
import re

def clean_currency(x, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + 'null_values_encountered',
        dirty_column_name + 'leading_trailing_whitespace',
        dirty_column_name + 'commas_replaced',
        dirty_column_name + 'currency_symbol_replaced'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + 'null_values_encountered'] += 1
        return None

    # Remove leading and trailing whitespace
    original_x = x
    x = x.strip()
    if x != original_x:
        df_report.at[0, dirty_column_name + 'leading_trailing_whitespace'] += 1

    # Replace commas with dots
    original_x = x
    x = x.replace(',', '.')
    if x != original_x:
        df_report.at[0, dirty_column_name + 'commas_replaced'] += 1

    # Define a regex pattern to match the currency symbol or code
    currency_pattern = r'(\$|€|¥|£)'

    # Extract the currency symbol from the payment
    match = re.search(currency_pattern, x)
    if match:
        df_report.at[0, dirty_column_name + 'currency_symbol_replaced'] += 1
        currency_symbol = match.group()
        if currency_symbol == '$':
            currency_code = 'USD'
        elif currency_symbol == '€':
            currency_code = 'EUR'
        elif currency_symbol == '¥':
            currency_code = 'JPY'
        elif currency_symbol == '£':
            currency_code = 'GBP'

        # Remove the currency symbol from the payment
        x = x.replace(currency_symbol, '')
    else:
        currency_code = None

    # Append the currency code to the payment
    if currency_code:
        x += ' ' + currency_code

    return x