"""Module containing functions for data cleaning."""

import pandas as pd
import re
from dateutil.parser import parse  # <-- For dates

# Important assumption:
#   The functions below contain a counter for all the modifications they applied.
#   This is usually in the form of "cleaning_summary_date"
#   1. This counter needs to be declared before running the function.
#   2. The counter needs to be resetted somewhere else.

#########################################
# Function for cleaning DATES
#########################################
def clean_dates(x):
    #global cleaning_summary_date
    
    # Skip Null values
    if pd.isna(x):
        cleaning_summary_date["null_values_encountered"] += 1
        return None
    try:
        # Attempt to parse the date using dateutil.parser.parse
        date_obj = parse(x, fuzzy=True)
        cleaning_summary_date["date_parsing_success"] += 1
        return date_obj.strftime('%Y-%m-%d')  # Convert date to YYYY-MM-DD format
    except Exception as e:
        cleaning_summary_date["date_parsing_failed"] += 1
        return None  # Return None if parsing fails

#########################################
# Function for cleaning PHONES
#########################################
def clean_phone(phone_number):
    #global cleaning_summary_phone
    
    # Skip Null values
    if pd.isna(date_str):
        cleaning_summary_phone["null_values_encountered"] += 1
        return None
    
    # Regular expression pattern to match valid phone numbers
    pattern = r'\+?[0-9]+(?:\s*[\-()x.]?\s*[0-9]+)*'
    
    # Find all phone number matches in the input string
    matches = re.findall(pattern, phone_number)
    
    # If no matches found, return None
    if not matches:
        return None
    
    # Select the first match as the cleaned phone number
    cleaned_phone_number = matches[0]
    
    # Remove non-numeric characters
    cleaned_phone_number = re.sub(r'\D', '', cleaned_phone_number)
    
    # Check if the cleaned phone number has a valid length
    if len(cleaned_phone_number) < 10 or len(cleaned_phone_number) > 15:
        return None
    
    # Add country code if missing
    if len(cleaned_phone_number) == 10:
        cleaned_phone_number = '+1' + cleaned_phone_number
    
    return cleaned_phone_number

#########################################
# Function for cleaning GEOMETRICAL data 
#########################################
def clean_geo(x):
    #global cleaning_summary_geo 
    
    # Skip Null values
    if pd.isna(x):
        cleaning_summary_geo["null_values_encountered"] += 1
        return None

    # Whitespace stripping
    original_x = x
    x = x.strip()
    if x != original_x:
        cleaning_summary_geo["leading_trailing_whitespace"] += 1
    
    # Trying to convert to float, except returning None if it fails
    try:
        original_x = x
        x = float(x)
        if x != original_x:
            cleaning_summary_geo["converted_to_float"] += 1
    except ValueError:
        cleaning_summary_geo["ValueError --> None"] += 1
        return None

    #fails to convert to float, they remain a string, and none
    # Check if it's a string
    if isinstance(x, str):
        cleaning_summary_geo["string --> None"] += 1
        return None
    
    
    if not (-90 <= x <= 90):
        cleaning_summary_geo["Number out of scope"] += 1
        return None
    
    return x

#########################################
# Function for cleaning EMAIL addresses
#########################################

def clean_email(x):
    #global cleaning_summary_email
    
    # Skip Null values
    if pd.isna(x):
        cleaning_summary_email["null_values_encountered"] += 1
        return None
    
    # Remove leading and trailing whitespace
    original_x = x
    x = x.strip()
    if x != original_x:
        cleaning_summary_email["leading_trailing_whitespace"] += 1
    
    # Remove spaces in the domain name
    original_x = x
    x = re.sub(r'\s+', '', x)
    if x != original_x:
        cleaning_summary_email["spaces_in_domain_name"] += 1
    
    # Remove double periods in the domain name
    original_x = x
    x = re.sub(r'\.{2,}', '.', x)
    if x != original_x:
        cleaning_summary_email["double_periods_in_domain_name"] += 1
    
    # Remove double @ symbols
    original_x = x
    x = re.sub(r'@{2,}', '@', x)
    if x != original_x:
        cleaning_summary_email["double_at_symbols"] += 1
    
    # Remove hyphens at the beginning or end of the domain name
    original_x = x
    x = re.sub(r'(?<!\.)-|-(?![^.])', '', x)
    if x != original_x:
        cleaning_summary_email["hyphens_in_domain_name"] += 1
    
    # Check if the email matches the basic pattern
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]{2,}$', x):
        cleaning_summary_email["invalid_email_pattern"] += 1
        return None
    else:
        cleaning_summary_email["valid_emails"] += 1

    return x


#########################################
# Function for cleaning PAYMENT data
#########################################


def clean_payment(x):
    global cleaning_summary_currency
    
    # Skip Null values
    if pd.isna(x):
        cleaning_summary_currency["null_values_encountered"] += 1
        return None
    
    # Remove leading and trailing whitespace
    original_x = x
    x = x.strip()
    if x != original_x:
        cleaning_summary_currency["leading_trailing_whitespace"] += 1
    
    # Replace commas with dots
    original_x = x
    x = x.replace(',', '.')
    if x != original_x:
        cleaning_summary_currency["commas_replaced"] += 1

    
    # Define a regex pattern to match the currency symbol or code
    currency_pattern = r'(\$|€|¥|£)'
    
    # Extract the currency symbol from the payment
    match = re.search(currency_pattern, x)
    if match:
        cleaning_summary_currency["currency_symbol_replaced"] += 1
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

#########################################
# Function for cleaning EXTRACTING currency
#########################################

def extract_currency(payment):
    if pd.isna(payment):
        return None
    
    # Define a regex pattern to match the currency code at the end of the payment string
    currency_pattern = r'\b[A-Z]{3}\b'
    
    # Extract the currency code from the payment
    match = re.search(currency_pattern, payment)
    if match:
        currency_code = match.group()
    else:
        currency_code = None
    
    return currency_code

#########################################
# Function for cleaning BOOLEAN
#########################################

def clean_boolean(x):
    global cleaning_summary_bool
    
    # Skip Null values
    if pd.isna(x):
        cleaning_summary_bool["null_values_encountered"] += 1
        return None
    
    true_like = ['1', 'yes', 'true', 'y']
    false_like = ['0', 'no', 'false', 'n']
    
    if isinstance(x, str):
        value_lower = x.lower()
        if value_lower in true_like:
            cleaning_summary_bool["trues_converted"] += 1
            return True
        elif value_lower in false_like:
            cleaning_summary_bool["falses_converted"] += 1
            return False
    return None

#########################################
# Function for cleaning NAME
#########################################

def clean_name(x):
    return None