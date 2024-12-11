import pandas as pd
import re

def clean_phone(phone_number, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_null_values_encountered',
        dirty_column_name + '_valid_phone_numbers',
        dirty_column_name + '_invalid_phone_numbers'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(phone_number):  # Skip Null values
        df_report.at[0, dirty_column_name + '_null_values_encountered'] += 1
        return None

    # Regular expression pattern to match valid phone numbers
    pattern = r'\+?[0-9]+(?:\s*[\-()x.]?\s*[0-9]+)*'

    # Find all phone number matches in the input string
    matches = re.findall(pattern, phone_number)

    # If no matches found, increment the counter for invalid_phone_numbers and return None
    if not matches:
        df_report.at[0, dirty_column_name + '_invalid_phone_numbers'] += 1
        return None

    # Select the first match as the cleaned phone number
    cleaned_phone_number = matches[0]

    # Remove non-numeric characters
    cleaned_phone_number = re.sub(r'\D', '', cleaned_phone_number)

    # Check if the cleaned phone number has a valid length
    if len(cleaned_phone_number) < 10 or len(cleaned_phone_number) > 15:
        df_report.at[0, dirty_column_name + '_invalid_phone_numbers'] += 1
        return None

    # Increment the counter for valid_phone_numbers
    df_report.at[0, dirty_column_name + '_valid_phone_numbers'] += 1

    # Add country code if missing
    if len(cleaned_phone_number) == 10:
        cleaned_phone_number = '+1' + cleaned_phone_number

    return cleaned_phone_number