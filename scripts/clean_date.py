import pandas as pd
import re
from dateutil.parser import parse             # <-- For dates
from .update_report import update_report      # <-- For a more elegant clean_dates()

# Function to clean dates without reporting
def clean_date_noreport(x):
    """ The core function for cleaning dates without in-build reporting.
    Usage example: df['cleaned_dates'] = df['dirty_dates'].apply(clean_dates_noreport)."""
    if pd.isna(x):                            # Skip Null values
        return None
    try:
        date_obj = parse(x, fuzzy=True)       # Attempt to parse the date using dateutil.parser.parse
        return date_obj.strftime('%Y-%m-%d')  # Convert date to YYYY-MM-DD format
    except Exception as e:      
        return None                           # Return None if parsing fails


    
def clean_date(x, df_report, dirty_column_name):
    # Initialize columns if they don't exist
    columns_to_initialize = [
        dirty_column_name + '_nulls_encountered',
        dirty_column_name + '_parsing_success',
        dirty_column_name + '_parsing_failed'
    ]
    for column in columns_to_initialize:
        if column not in df_report.columns:
            df_report[column] = 0

    # Increment appropriate counters based on the value of x
    if pd.isna(x):  # Skip Null values
        df_report.at[0, dirty_column_name + '_nulls_encountered'] += 1
        return None

    try:
        date_obj = parse(x, fuzzy=True)  # Attempt to parse the date using dateutil.parser.parse
        df_report.at[0, dirty_column_name + '_parsing_success'] += 1
        return date_obj.strftime('%Y-%m-%d')  # Convert date to YYYY-MM-DD format

    except Exception as e:
        df_report.at[0, dirty_column_name + '_parsing_failed'] += 1
        return None    

# Function to clean dates with reporting
def clean_date_updatereport(x, df_report, dirty_column_name):
    """ A cleaning function for dates with reporting. 
    Uses a helper function to fill an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (dirty_column_name).
    Usage example: df['cleaned_dates'] = df['dirty_dates'].apply(clean_dates, df_report=df_report, dirty_column_name='dirty_dates')
    """
    if pd.isna(x):                            # Skip Null values
        update_report(df_report, dirty_column_name + '_nulls_encountered', dirty_column_name)
        return None
    
    try:
        date_obj = parse(x, fuzzy=True)       # Attempt to parse the date using dateutil.parser.parse
        update_report(df_report, dirty_column_name + '_parsing_success', dirty_column_name)
        return date_obj.strftime('%Y-%m-%d')  # Convert date to YYYY-MM-DD format
    
    except Exception as e:
        update_report(df_report, dirty_column_name + '_parsing_failed', dirty_column_name)
        return None                           # Return None if parsing fails
    
    
# Function to clean dates with reporting
def clean_date_flat(x, df_report, dirty_column_name):
    """ A cleaning function for dates with reporting. 
    Does not use a helper function to fill an empty df (df_report) with statistics of what modifications this function made.
    Keeps track of which column the statistics was gathered from (dirty_column_name).
    Usage example: df['cleaned_dates'] = df['dirty_dates'].apply(clean_dates, df_report=df_report, dirty_column_name='dirty_dates')
    """
    if pd.isna(x):                                 # Skip Null values
        
        report_column_name1 = dirty_column_name + '_nulls_encountered'
        if report_column_name1 not in df_report.columns: # If the column doesn't exist, create it with initial value 0
            df_report[report_column_name1] = 0     
        if 0 not in df_report.index:               # Initialize the value if the index '0' is not present
            df_report.loc[0] = 0
        df_report.at[0, report_column_name1] += 1  # Update the value in the existing column
        
        return None
    
    try:
        date_obj = parse(x, fuzzy=True)            # Attempt to parse the date using dateutil.parser.parse
        
        report_column_name2 = dirty_column_name + '_parsing_success'
        if report_column_name2 not in df_report.columns:  #If the column doesn't exist, create it with initial value 0
            df_report[report_column_name2] = 0    
        if 0 not in df_report.index:
            df_report.loc[0] = 0
        df_report.at[0, report_column_name2] += 1  # Update the value in the existing column
        
        return date_obj.strftime('%Y-%m-%d')       # Convert date to YYYY-MM-DD format
    
    except Exception as e:
        
        report_column_name3 = dirty_column_name + '_parsing_failed'
        if report_column_name3 not in df_report.columns:   # If the column doesn't exist, create it with initial value 0
            df_report[report_column_name3] = 0     
        if 0 not in df_report.index:
            df_report.loc[0] = 0
        df_report.at[0, report_column_name3] += 1  # Update the value in the existing column
        
        return None                                # Return None if parsing fails

