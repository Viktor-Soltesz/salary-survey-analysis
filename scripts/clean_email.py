import pandas as pd
import re

def clean_email(x, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_null_values_encountered',
        dirty_column_name + '_leading_trailing_whitespace',
        dirty_column_name + '_spaces_in_domain_name',
        dirty_column_name + '_double_periods_in_domain_name',
        dirty_column_name + '_double_at_symbols',
        dirty_column_name + '_hyphens_in_domain_name',
        dirty_column_name + '_invalid_email_pattern',
        dirty_column_name + '_valid_emails'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + '_null_values_encountered'] += 1
        return None

    # Remove leading and trailing whitespace
    original_x = x
    x = x.strip()
    if x != original_x:
        df_report.at[0, dirty_column_name + '_leading_trailing_whitespace'] += 1

    # Remove spaces in the domain name
    original_x = x
    x = re.sub(r'\s+', '', x)
    if x != original_x:
        df_report.at[0, dirty_column_name + '_spaces_in_domain_name'] += 1

    # Remove double periods in the domain name
    original_x = x
    x = re.sub(r'\.{2,}', '.', x)
    if x != original_x:
        df_report.at[0, dirty_column_name + '_double_periods_in_domain_name'] += 1

    # Remove double @ symbols
    original_x = x
    x = re.sub(r'@{2,}', '@', x)
    if x != original_x:
        df_report.at[0, dirty_column_name + '_double_at_symbols'] += 1

    # Remove hyphens at the beginning or end of the domain name
    original_x = x
    x = re.sub(r'(?<!\.)-|-(?![^.])', '', x)
    if x != original_x:
        df_report.at[0, dirty_column_name + '_hyphens_in_domain_name'] += 1

    # Check if the email matches the basic pattern
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}$', x):
        df_report.at[0, dirty_column_name + '_invalid_email_pattern'] += 1
        return None
    else:
        df_report.at[0, dirty_column_name + '_valid_emails'] += 1

    return x