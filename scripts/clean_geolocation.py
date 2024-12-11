import pandas as pd
import re

def clean_geo(x, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_null_values_encountered',
        dirty_column_name + '_leading_trailing_whitespace',
        dirty_column_name + '_converted_to_float',
        dirty_column_name + '_ValueError --> None',
        dirty_column_name + '_string --> None',
        dirty_column_name + '_Number out of scope'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + '_null_values_encountered'] += 1
        return None

    # Whitespace stripping
    original_x = x
    x = x.strip()
    if x != original_x:
        df_report.at[0, dirty_column_name + '_leading_trailing_whitespace'] += 1

    # Trying to convert to float, except returning None if it fails
    try:
        original_x = x
        x = float(x)
        if x != original_x:
            df_report.at[0, dirty_column_name + '_converted_to_float'] += 1
    except ValueError:
        df_report.at[0, dirty_column_name + '_ValueError --> None'] += 1
        return None

    # Check if it's a string
    if isinstance(x, str):
        df_report.at[0, dirty_column_name + '_string --> None'] += 1
        return None

    if not (-90 <= x <= 90):
        df_report.at[0, dirty_column_name + '_Number out of scope'] += 1
        return None

    return x