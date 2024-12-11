import pandas as pd
import re

def search_single_keyword(df, search_column, new_column, keyword, key_value):
    """
    Searches for keywords in a specified column and fills a new column with a key-value pair where there is a match.

    Parameters:
    df (pd.DataFrame): The dataframe to operate on.
    search_column (str): The name of the column to search the keywords in.
    new_column (str): The name of the new column to insert the key-value pair into.
    keyword (str): A keyword to search for.
    key_value (str): The value to insert into the new column where a keyword match is found.

    Returns:
    pd.DataFrame: The modified dataframe with the new column added.
    """
    keyword = keyword.lower()
    
    # Create the new column with the key_value where the pattern matches, else NaN
    df[new_column] = df[search_column].apply(lambda x: key_value if pd.notnull(x) and keyword in x.lower() else None)
    
    return df

def search_double_keyword(df, search_column, new_column, keyword1, keyword2, key_value):
    """
    Searches for two keywords in a specified column and fills a new column with a key-value pair where both keywords are present.

    Parameters:
    df (pd.DataFrame): The dataframe to operate on.
    search_column (str): The name of the column to search the keywords in.
    new_column (str): The name of the new column to insert the key-value pair into.
    keyword1 (str): The first keyword to search for.
    keyword2 (str): The second keyword to search for.
    key_value (str): The value to insert into the new column where both keyword matches are found.

    Returns:
    pd.DataFrame: The modified dataframe with the new column added.
    """
    # Convert keywords to lowercase for case-insensitive comparison
    keyword1 = keyword1.lower()
    keyword2 = keyword2.lower()
    
    # Create the new column with the key_value where both keywords are present, else NaN
    df[new_column] = df.apply(lambda row: key_value if pd.isnull(row[new_column]) and keyword1 in row[search_column].lower() and keyword2 in row[search_column].lower() else row[new_column], axis=1)
    
    return df

def search_double_keyword_obs(df, search_column, new_column, keyword1, keyword2, key_value):
    """
    Searches for two keywords in a specified column and fills a new column with a key-value pair where both keywords are present.

    Parameters:
    df (pd.DataFrame): The dataframe to operate on.
    search_column (str): The name of the column to search the keywords in.
    new_column (str): The name of the new column to insert the key-value pair into.
    keyword1 (str): The first keyword to search for.
    keyword2 (str): The second keyword to search for.
    key_value (str): The value to insert into the new column where both keyword matches are found.

    Returns:
    pd.DataFrame: The modified dataframe with the new column added.
    """
    # Convert keywords to lowercase for case-insensitive comparison
    keyword1 = keyword1.lower()
    keyword2 = keyword2.lower()
    
    # Create the new column with the key_value where both keywords are present, else NaN
    df[new_column] = df[search_column].apply(
        lambda x: key_value if pd.notnull(x) and keyword1 in x.lower() and keyword2 in x.lower() else None
    )
    
    return df
