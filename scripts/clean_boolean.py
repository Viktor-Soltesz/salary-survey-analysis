import pandas as pd
import re

from .update_report import update_report 

def clean_boolean_noreport(x):
    if pd.isna(x): # Skip Null values
        return None

    true_like = ['1', 'yes', 'true', 'y']
    false_like = ['0', 'no', 'false', 'n']
    
    if isinstance(x, bool):  # Handle boolean input types directly
        return x
    
    if isinstance(x, str):
        value_lower = x.lower()
        if value_lower in true_like:
            return True
        elif value_lower in false_like:
            return False
        else: # strings encountered that cannot be converted
            return None
        
    else: # non-string data type encountered that cannot be converted
        return None

    
def clean_boolean(x, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_nulls_encountered',
        dirty_column_name + '_natural_bools_untouched',
        dirty_column_name + '_truelike_converted',
        dirty_column_name + '_falselike_converted',
        dirty_column_name + '_strings_nonconverted',
        dirty_column_name + '_nonstrings_nonconverted'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + '_nulls_encountered'] += 1
        return None

    true_like = ['1', 'yes', 'true', 'y']
    false_like = ['0', 'no', 'false', 'n']

    if isinstance(x, bool):  # Handle boolean input types directly
        df_report.at[0, dirty_column_name + '_natural_bools_untouched'] += 1
        return x

    if isinstance(x, str):
        value_lower = x.lower()
        if value_lower in true_like:
            df_report.at[0, dirty_column_name + '_truelike_converted'] += 1
            return True
        elif value_lower in false_like:
            df_report.at[0, dirty_column_name + '_falselike_converted'] += 1
            return False
        else:  # strings encountered that cannot be converted
            df_report.at[0, dirty_column_name + '_strings_nonconverted'] += 1
            return None

    # non-string data type encountered that cannot be converted
    df_report.at[0, dirty_column_name + '_nonstrings_nonconverted'] += 1
    return None





def clean_boolean_elegant(x, df_report, dirty_column_name):
    if pd.isna(x): # Skip Null values
        update_report(df_report, dirty_column_name + '_nulls_encountered', dirty_column_name)
        return None
    
    true_like = ['1', 'yes', 'true', 'y']
    false_like = ['0', 'no', 'false', 'n']
    
    if isinstance(x, bool):  # Handle boolean input types directly
        update_report(df_report, dirty_column_name + '_natural_bools_untouched', dirty_column_name)
        return x
    
    if isinstance(x, str):
        value_lower = x.lower()
        if value_lower in true_like:
            update_report(df_report, dirty_column_name + '_truelike_converted', dirty_column_name)
            return True
        elif value_lower in false_like:
            update_report(df_report, dirty_column_name + '_falselike_converted', dirty_column_name)
            return False
        else: # strings encountered that cannot be converted
            update_report(df_report, dirty_column_name + '_strings_nonconverted', dirty_column_name)
            return None
        
    else: # non-string data type encountered that cannot be converted
        update_report(df_report, dirty_column_name + '_nonstrings_nonconverted', dirty_column_name)
        return None
