"""Module containing functions for testing"""

import pandas as pd

# The function to create test-cases
def create_dirty_and_expected_data(df, dirty_column, expected_column, dirty_data_list, expected_data_list):
    #Input arguments: 
        # 1. a DataFrame which the test data will be appended to, 
        # 2. name for the 'Dirty' column (str), 
        # 3. name for the 'Expected' column (str), 
        # 4. dirty data (list), 
        # 5. expected data (list)
    # Important assumption: df initially needs to be longer than the cases we define later
    # we define some test-cases with expected values pairs, and the rest of the df will be filled with Nones.
    df[dirty_column] = dirty_data_list + [None] * (len(df) - len(dirty_data_list))
    df[expected_column] = expected_data_list + [None] * (len(df) - len(expected_data_list))
    

# The function for applying any cleaning function, then comparing the result with the expected values
def test_cleaning(clean_function, df, dirty_column, expected_column):
    #Input arguments: 
        # 1. The cleaning function that will be tested, 
        # 2. the Dataframe containing the dirty & expected variables       
        # 3. name for the 'Dirty' column (str), 
        # 4. name for the 'Expected' column (str)
        
    # Apply the cleaning function to the dirty column, and store the result in a new generic "cleaned_data" columns.
    cleaned_data = df[dirty_column].apply(clean_function)
    
    # Compare the cleaned data with the expected data
    comparison_result = cleaned_data == df[expected_column]
    
    # Print the comparison result
    print("Comparison Result:")
    for idx, (cleaned_value, expected_value) in enumerate(zip(cleaned_data, df[expected_column])):
        if cleaned_value != expected_value:
            print(f"(Row {idx}) Cleaned Value: {cleaned_value} {type(cleaned_value)}, Expected Value: {expected_value} {type(expected_value)}, ")
    
    return comparison_result.all()


# The function for applying any cleaning function, then comparing the result with the expected values
def test_cleaning_report(clean_function, df, df_report, dirty_column, expected_column):
    #Input arguments: 
        # 1. The cleaning function that will be tested, 
        # 2. the Dataframe containing the dirty & expected variables
        # 3. df_report
        # 4. name for the 'Dirty' column (str), 
        # 5. name for the 'Expected' column (str)
        
    # Apply the cleaning function to the dirty column, and store the result in a new generic "cleaned_data" columns.
    # cleaned_data = df[dirty_column].apply(clean_function, df_report=df_report, dirty_column_name=dirty_column) # Keyword-Argument-type definition
    cleaned_data = df[dirty_column].apply(clean_function, df_report, dirty_column) # Argument position-type definition
    
    # Compare the cleaned data with the expected data
    comparison_result = cleaned_data == df[expected_column]
    
    # Print the comparison result
    print("Comparison Result:")
    for idx, (cleaned_value, expected_value) in enumerate(zip(cleaned_data, df[expected_column])):
        if cleaned_value != expected_value:
            print(f"(Row {idx}) Cleaned Value: {cleaned_value} {type(cleaned_value)}, Expected Value: {expected_value} {type(expected_value)}, ")
    
    return comparison_result.all()


def test_cleaning_report_2(clean_function, df, df_report, prefix, expected_column):
    """
    A function for applying the original clean_salary function, then comparing the result with the expected values.

    Parameters:
    - clean_function: The cleaning function that will be tested (original clean_salary function).
    - df: The DataFrame containing the dirty & expected variables.
    - df_report: The DataFrame for reporting.
    - prefix: Prefix for the reporting columns (str).
    - expected_column: Name for the 'Expected' column (str).

    Returns:
    - bool: True if all cleaned values match the expected values, False otherwise.
    """
    # Apply the cleaning function to the dirty column
    cleaned_data = df[prefix].apply(clean_function, df_report, prefix)
    
    # Compare the cleaned data with the expected data
    comparison_result = True
    for idx, (cleaned_value, expected_value) in enumerate(zip(cleaned_data, df[expected_column])):
        if cleaned_value != expected_value:
            comparison_result = False
            print(f"(Row {idx}) Cleaned Value: {cleaned_value} {type(cleaned_value)}, Expected Value: {expected_value} {type(expected_value)}, ")
    
    return comparison_result

def test_cleaning_report_3(clean_function, df, df_report, prefix, expected_column):
    """
    A function for applying the original clean_salary function, then comparing the result with the expected values.

    Parameters:
    - clean_function: The cleaning function that will be tested (original clean_salary function).
    - df: The DataFrame containing the dirty & expected variables.
    - df_report: The DataFrame for reporting.
    - prefix: Prefix for the reporting columns (str).
    - expected_column: Name for the 'Expected' column (str).

    Returns:
    - bool: True if all cleaned values match the expected values, False otherwise.
    """
    # Apply the cleaning function to the dirty column
    cleaned_data = df[prefix].apply(clean_function, df_report, prefix)
    
    # Compare the cleaned data with the expected data
    comparison_result = True
    for idx, (cleaned_value, expected_value) in enumerate(zip(cleaned_data, df[expected_column])):
        if cleaned_value != expected_value:
            comparison_result = False
            print(f"(Row {idx}) Cleaned Value: {cleaned_value} {type(cleaned_value)}, Expected Value: {expected_value} {type(expected_value)}, ")
    
    return comparison_result

