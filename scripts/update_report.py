import pandas as pd

def update_report(df_report, column_name, dirty_column_name):
    if column_name not in df_report.columns:
        df_report[column_name] = 0
    if 0 not in df_report.index:
        df_report.loc[0] = 0
    df_report.at[0, column_name] += 1