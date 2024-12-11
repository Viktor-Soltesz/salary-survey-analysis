import pandas as pd
import re

def clean_salary(x, df_report, prefix):
    """ A cleaning function for salary with reporting. 
    Fills an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (prefix).
    Usage example: df['cleaned_salary'] = df['dirty_salary'].apply(clean_salary, df_report=df_report, prefix='mydf_dirty_salary')
    
    Tip: Should be used before pd.to_numeric()
    Tip: Handling none-s should be done separately, this function skips nones.
    """
    # Initialize columns if they don't exist
    columns_to_initialize = [
        prefix + '_null_values_encountered',
        prefix + '_strings_encountered',
        prefix + '_leading_trailing_whitespace',
        prefix + '_commas_replaced'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, prefix + '_null_values_encountered'] += 1
        return None

    # Check for strings
    if isinstance(x, str):
        df_report.at[0, prefix + '_strings_encountered'] += 1
        
        # Remove leading and trailing whitespace
        original_x = x
        x = x.strip()
        if x != original_x:
            df_report.at[0, prefix + '_leading_trailing_whitespace'] += 1

        # Replace commas with dots. 
        # pd.to_numeric function can correctly identify '.' as decimal separators, but not ','
        original_x = x
        x = x.replace(',', '.')
        if x != original_x:
            df_report.at[0, prefix + '_commas_replaced'] += 1
        return x  # Return the cleaned string
    else:
        return x
    
def clean_salary_0616(x, df_report, prefix):
    """ A cleaning function for salary with reporting. 
    Fills an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (prefix).
    Usage example: df['cleaned_salary'] = df['dirty_salary'].apply(clean_salary, df_report=df_report, prefix='mydf_dirty_salary')
    
    Tip: Should be used before pd.to_numeric()
    Tip: Handling none-s should be done separately, this function skips nones.
    """
    # Initialize columns if they don't exist
    columns_to_initialize = [
        prefix + '_null_values_encountered',
        prefix + '_strings_encountered',
        prefix + '_leading_trailing_whitespace',
        prefix + '_+-_characters_removed',
        prefix + '_commas_replaced'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, prefix + '_null_values_encountered'] += 1
        return None

    # Check for strings
    if isinstance(x, str):
        df_report.at[0, prefix + '_strings_encountered'] += 1
        
        # Remove leading and trailing whitespace
        original_x = x
        x = x.strip()
        if x != original_x:
            df_report.at[0, prefix + '_leading_trailing_whitespace'] += 1
        
        # Remove weird numeric characters.
        original_x = x
        x = x.replace('+', '')
        x = x.replace('-', '')
        if x != original_x:
            df_report.at[0, prefix + '_+-_characters_removed'] += 1

        # Replace commas with dots. 
        # pd.to_numeric function can correctly identify '.' as decimal separators, but not ','
        original_x = x
        x = x.replace(',', '.')
        if x != original_x:
            df_report.at[0, prefix + '_commas_replaced'] += 1
        return x  # Return the cleaned string
    else:
        return x
    

def clean_salary_0615(x, df_report, prefix):
    """ A cleaning function for salary with reporting. 
    Fills an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (prefix).
    Usage example: df['cleaned_salary'] = df['dirty_salary'].apply(clean_salary, df_report=df_report, prefix='mydf_dirty_salary')
    
    Tip: Should be used before pd.to_numeric()
    Tip: Handling none-s should be done separately, this function skips nones.
    """
    # Initialize columns if they don't exist
    columns_to_initialize = [
        prefix + '_null_values_encountered',
        prefix + '_strings_encountered',
        prefix + '_leading_trailing_whitespace',
        prefix + '_commas_replaced'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, prefix + '_null_values_encountered'] += 1
        return None

    # Check for strings
    if isinstance(x, str):
        df_report.at[0, prefix + '_strings_encountered'] += 1
        
        # Remove leading and trailing whitespace
        original_x = x
        x = x.strip()
        if x != original_x:
            df_report.at[0, prefix + '_leading_trailing_whitespace'] += 1

        # Replace commas with dots. 
        # pd.to_numeric function can correctly identify '.' as decimal separators, but not ','
        original_x = x
        x = x.replace(',', '.')
        if x != original_x:
            df_report.at[0, prefix + '_commas_replaced'] += 1
        return x  # Return the cleaned string
    else:
        return x

def clean_salary_2(x, df_report, dirty_column_name):
    """ A cleaning function for salary with reporting. 
    Fills an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (prefix).
    Usage example: df['cleaned_salary'] = df['dirty_salary'].apply(clean_salary, df_report=df_report, prefix='mydf_dirty_salary')
    
    Tip: Should be used before pd.to_numeric()
    Tip: Handling none-s should be done separately, this function skips nones.
    """
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_null_values_encountered',
        dirty_column_name + '_strings_encountered',
        dirty_column_name + '_leading_trailing_whitespace',
        dirty_column_name + '_commas_replaced'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + '_null_values_encountered'] += 1
        return None

    # Check for strings
    if isinstance(x, str):
        df_report.at[0, dirty_column_name + '_strings_encountered'] += 1
        
        # Remove leading and trailing whitespace
        original_x = x
        x = x.strip()
        if x != original_x:
            df_report.at[0, dirty_column_name + '_leading_trailing_whitespace'] += 1

        # Replace commas with dots. 
        # pd.to_numeric function can correctly identify '.' as decimal separators, but not ','
        original_x = x
        x = x.replace(',', '.')
        if x != original_x:
            df_report.at[0, dirty_column_name + '_commas_replaced'] += 1
        return x  # Return the cleaned string
    else:
        return x