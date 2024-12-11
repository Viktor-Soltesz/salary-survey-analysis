import pandas as pd

def memory_summary(df, num_rows=None):
    """ A simple function to list out the memory usage of each column.
    Inputs: (df, an integer to control how many columns should be printed)"""
    # Calculate Total memory usage:
    print("Total memory Usage:")
    print(f"\t{round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB")
    
    # Calculate memory usage for each column
    memory_usage_per_column = {column: round(df[column].memory_usage(deep=True) / (1024 * 1024), 2) for column in df.columns}
    # Sort columns by memory usage in descending order
    sorted_columns_by_memory = sorted(memory_usage_per_column.items(), key=lambda x: x[1], reverse=True)
    
    # Print detailed memory usage in descending order
    print(f"Detailed memory Usage [{len(sorted_columns_by_memory)} columns] (descending order):")
    if num_rows is None:
        num_rows = len(sorted_columns_by_memory)
    for i in range(min(num_rows, len(sorted_columns_by_memory))):
        column, memory_usage = sorted_columns_by_memory[i]
        print(f"\t{column}: {memory_usage} MB")