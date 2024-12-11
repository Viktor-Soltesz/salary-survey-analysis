#!/usr/bin/env python
# coding: utf-8

# <font size="6.5"><b>Cleaning Salaries data</b></font>

# <h1 style="background-color: #0e2e3b; color: white; font-size: 40px; font-weight: bold; padding: 10px;"> Import libraries & data, general settings </h1>

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import re

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from scipy.stats import levene, shapiro

# If the notebook is opened from the "notebooks" folder, we need to append the main directory to the "python path" so it sees all subfolders.
import sys
sys.path.append('../')


# In[4]:


df_it18_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2018.csv', low_memory=False)
df_it19_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2019.csv', low_memory=False)
df_it20_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2020.csv', low_memory=False)
df_it21_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2021.csv', low_memory=False)
df_it22_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2022.csv', low_memory=False)
df_it23_ini = pd.read_csv('../data/raw/IT_Salary_Survey_EU_2023.csv', low_memory=False)

df_k19_ini = pd.read_csv('../data/raw/kaggle_survey_2019_responses.csv', low_memory=False)
df_k20_ini = pd.read_csv('../data/raw/kaggle_survey_2020_responses.csv', low_memory=False)
df_k21_ini = pd.read_csv('../data/raw/kaggle_survey_2021_responses.csv', low_memory=False)
df_k22_ini = pd.read_csv('../data/raw/kaggle_survey_2022_responses.csv', low_memory=False)

df_ai_ini = pd.read_csv('../data/raw/ai-jobsnet_salaries_2024.csv', low_memory=False)


# In[5]:


country_salary_stats = pd.read_csv('../data/world_economic_indices/country_salary_stats.csv', sep=';', low_memory=False)


# In[6]:


dfs_ini = [
    df_it18_ini,
    df_it19_ini,
    df_it20_ini,
    df_it21_ini,
    df_it22_ini,
    df_it23_ini,
    df_k19_ini,
    df_k20_ini,
    df_k21_ini,
    df_k22_ini,
    df_ai_ini,
    country_salary_stats]


# In[7]:


len_it18_ini = len(df_it18_ini)
len_it19_ini = len(df_it19_ini)
len_it20_ini = len(df_it20_ini)
len_it21_ini = len(df_it21_ini)
len_it22_ini = len(df_it22_ini)
len_it23_ini = len(df_it23_ini)

len_k19_ini = len(df_k19_ini)
len_k20_ini = len(df_k20_ini)
len_k21_ini = len(df_k21_ini)
len_k22_ini = len(df_k22_ini)

len_ai_ini = len(df_ai_ini)


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> Styles </h2>

# In[9]:


# General Display settings

# Column display is supressed by default
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#changing the display format
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Plotting format
#print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')


# In[10]:


""" Header formatting
    # H1
<h1 style="background-color: #0e2e3b; color: white; font-size: 40px; font-weight: bold; padding: 10px;"> TEST TEST</h1>

    # H2 
<h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> TEST TEST</h2>

    # H3
<h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> TEST TEST</h3>

    # H4
<h4 style="background-color: #0AB89E; color: white; font-size: 20px; font-weight: bold; padding: 5px;"> TEST TEST</h4>
""";


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;">  Basic standardization</h2>

# In[12]:


# Column names to lowercase
for df in dfs_ini:
    df.columns = df.columns.str.lower()

## Values to lowercase
#for i in range(len(dfs)):
#    dfs[i] = dfs[i].applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Whitespaces in column names
for df in dfs_ini:
    df.columns = df.columns.str.replace(' ', '_')


# In[13]:


df_it20_ini.head(2)


# In[14]:


# Every datapoints to lowercase

df_it18_ini =df_it18_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_it19_ini =df_it19_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_it20_ini =df_it20_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_it21_ini =df_it21_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_it22_ini =df_it22_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_it23_ini =df_it23_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_k19_ini = df_k19_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_k20_ini = df_k20_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_k21_ini = df_k21_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_k22_ini = df_k22_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df_ai_ini =df_ai_ini.applymap(lambda x: x.lower() if isinstance(x, str) else x)
country_salary_stats = country_salary_stats.applymap(lambda x: x.lower() if isinstance(x, str) else x)


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> Downcasting data types for better memory usage</h2>

# In[16]:


from scripts.memory_summary import memory_summary


# In[6]:


#?? memory_summary


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Kaggle </h3>

# In[19]:


memory_summary(df_k21_ini, 10)


# In[20]:


df_k21_ini = df_k21_ini.astype('category')


# In[21]:


memory_summary(df_k21_ini, 10)


# Categorical conversion made a significant impact on memory usage.\
# It's logical since the survey contained fixed-choice questions.\
# I proceed by converting all the remaining Kaggle datasets.

# In[23]:


df_k19_ini = df_k19_ini.astype('category')
df_k20_ini = df_k20_ini.astype('category')
#df_k21_ini = df_k21_ini.astype('category')
df_k22_ini = df_k22_ini.astype('category')


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;">  Germany IT survey</h3>

# In[25]:


memory_summary(df_it23_ini, 10)


# It already requires so little memory usage, that I do not proceed with downcasting

# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;">  AI-Jobs.net </h3>

# In[28]:


memory_summary(df_ai_ini, 10)


# It already requires so little memory usage, but I proceed with downcasting to have consistent variable types.

# In[30]:


df_ai_ini = df_ai_ini.astype('category')


# <h1 style="background-color: #0e2e3b; color: white; font-size: 40px; font-weight: bold; padding: 10px;"> Comprehending the data. Uniformization. </h1>

# Uniformization goal:\
#     - "salary": Yearly gross salary with bonuses included [int, in USD],\
#     - "year": Year in which the datapoint was recorded [int],\
#     - "job_title": The title of the job eg.: "Data Engineer"[str],\
#     - "job_group": An arbitrary grouping of similar job titles: "Data Engineer" & "Database Engineer" --> "DA" [str],\
#     - 

# In[33]:


# A simple function to list out the column names and the most frequent values.

def top_5_values_per_column(df):
    # Create an empty DataFrame to store the result
    result_df = pd.DataFrame(columns=['column_name', 'top_5_values'])

    # Iterate through each column
    for column in df.columns:
        # Find the five most common values in the column
        top_5_values = df[column].value_counts().head(5).index.tolist()

        # Create a DataFrame with the current column name and top 5 values
        data_to_append = pd.DataFrame({'column_name': [column], 'top_5_values': [top_5_values]})

        # Concatenate the new DataFrame with the result DataFrame
        result_df = pd.concat([result_df, data_to_append], ignore_index=True)

    return result_df


# ### Importing the Clean_Salary function

# In[35]:


# Reloading a module
import importlib
import sys
import scripts.clean_salary

# Add the parent directory of 'scripts' to the module search path
sys.path.append('../')

# Reload the module
importlib.reload(scripts.clean_salary)
from scripts.clean_salary import clean_salary_0616
df_report = pd.DataFrame(index=[0])


# In[4]:


#?? clean_salary_0616


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> AI-Jobs.net </h2>

# In[38]:


top_5_values_per_column(df_ai_ini)


# In[39]:


# Define column name mappings
column_mappings = {
    'work_year': 'year',
    'experience_level': 'seniority_level',
    'employment_type': 'employment_status',
    'employee_residence': 'country',
    'salary': 'salary_in_currency',
    'salary_in_usd': 'salary'
}

# Rename columns using the mappings
df_ai_u = df_ai_ini.rename(columns=column_mappings)
df_ai_u.head(5)


# In[40]:


df_ai_u['salary'] = df_ai_u['salary'].apply(clean_salary_0616, df_report=df_report, prefix='df_ai_salary')
print(df_report)


# In[41]:


df_ai_u['salary'].info()


# There are no Null values, so we do not need to drop them.

# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Exchange rates</h3>
# <b>EUR --> USD exchange ratio calculations </b>.
# 
# I'm using the ratios that AI-Jobs.net used, since I can calculate it reversely and reuse those rates.\
# The rates they used was a fixed rate for the entire year, and aligns with the "Average" exchange rates for that year available online.

# In[44]:


# Converting salary and salary_in_currency back to numeric
df_ai_u['salary'] = pd.to_numeric(df_ai_u['salary'], errors='coerce')
df_ai_u['salary_in_currency'] = pd.to_numeric(df_ai_u['salary_in_currency'], errors='coerce')

# Calculating the ratios
df_ai_u['ratio'] = df_ai_u['salary'] / df_ai_u['salary_in_currency']

#changing the display format
pd.set_option('display.float_format', lambda x: '%.6f' % x)

#listing out the calculated ratios. Within each year, they're almost the same, except the rounding error.
df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2020)].value_counts('ratio').head(10)


# Most probably they **rounded**  the values after conversion, and this resulted in the slightly different ratios above.\
# For methodology's sake I calculate the exchange ratio for USD columns and take their mean.

# In[46]:


eur2usd_2020 = df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2020)]['ratio'].mean()
eur2usd_2021 = df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2021)]['ratio'].mean()
eur2usd_2022 = df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2022)]['ratio'].mean()
eur2usd_2023 = df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2023)]['ratio'].mean()
eur2usd_2024 = df_ai_u[(df_ai_u['salary_currency'] == 'eur') & (df_ai_u['year'] == 2024)]['ratio'].mean()

# For other years that are not present in the AI-Jobs.net dataset, I use sources available online
eur2usd_2019 = 1.1199 # Source: https://www.exchangerates.org.uk/EUR-USD-spot-exchange-rates-history-2019.html
eur2usd_2018 = 1.1811 # Source: https://www.exchangerates.org.uk/EUR-USD-spot-exchange-rates-history-2018.html

#changing back the display format
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;">  Germany IT survey</h2>

# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2018 </h3>

# In[49]:


top_5_values_per_column(df_it18_ini)


# In[50]:


# Adding year and country
df_it18_ini['year'] = 2018
df_it18_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position': 'job_title',
    'years_of_experience': 'experience',
    'your_level': 'seniority_level',
    'current_salary': 'salary_eur',
    'main_language_at_work': 'language_at_work'
}

# Rename columns using the mappings
df_it18_u = df_it18_ini.rename(columns=column_mappings)
df_it18_u.head(2)


# In[51]:


# Cleaning report
df_it18_u['salary_eur'] = df_it18_u['salary_eur'].apply(clean_salary_0616, df_report=df_report, prefix='df_it18_salary')
df_report


# In[52]:


len_it18_salarydrop1 = len(df_it18_u) # For data quality report

# Comverting to numeric
df_it18_u['salary_eur'] = df_it18_u['salary_eur'].apply(pd.to_numeric, errors='raise')
# same appraoch: df_it18_u['salary_eur'] = pd.to_numeric(df_it18_u['salary_eur'], errors='raise')

# Dropping rows where salary is Null
df_it18_u.dropna(subset=['salary_eur'], inplace=True)

# Assigning 'int64' datatype. This serves as a self-check
df_it18_u['salary_eur'] = df_it18_u['salary_eur'].astype('int64')

# Create a new 'salary' column that will represent the salary in USD
df_it18_u['salary'] = df_it18_u['salary_eur'] * eur2usd_2018

len_it18_salarydrop2 = len(df_it18_u) # For data quality report

df_it18_u.head(2)


# In[53]:


df_it18_u[['salary_eur','salary']].info()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2019 </h3>

# In[55]:


top_5_values_per_column(df_it19_ini)


# In[56]:


# Adding year and country
df_it19_ini['year'] = 2019
df_it19_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position_(without_seniority)': 'job_title',
    'years_of_experience': 'experience',
    'your_main_technology_/_programming_language': 'skills',
    'yearly_brutto_salary_(without_bonus_and_stocks)': 'base_salary',
    'yearly_bonus': 'bonus',
    'yearly_stocks': 'stocks',
    'yearly_brutto_salary_(without_bonus_and_stocks)_one_year_ago._only_answer_if_staying_in_same_country': 'salary_1y_ago',
    'yearly_bonus_one_year_ago._only_answer_if_staying_in_same_country':'bonus_1y_ago',
    'yearly_stocks_one_year_ago._only_answer_if_staying_in_same_country':'stocks_1y_ago',
    'main_language_at_work': 'language_at_work'
}

# Rename columns using the mappings
df_it19_u = df_it19_ini.rename(columns=column_mappings)


# In[57]:


# Cleaning report

df_it19_u['base_salary'] = df_it19_u['base_salary'].apply(clean_salary_0616, df_report=df_report, prefix='df_it19_u_salary')
df_it19_u['bonus'] = df_it19_u['bonus'].apply(clean_salary_0616, df_report=df_report, prefix='df_it19_u_bonus')
df_it19_u['stocks'] = df_it19_u['stocks'].apply(clean_salary_0616, df_report=df_report, prefix='df_it19_u_stocks')
df_report


# There is a Null value in 'base_salary', that will need to be dropped. The cleaning function itself does not drop Null values, so I drop it separately.\
# There are hundreds of null values in 'bonus' and 'stocks', I'll convert them to zeroes.

# In[59]:


# Dropping values

len_it19_salarydrop1 = len(df_it19_u) # For data quality report

df_it19_u.dropna(subset=['base_salary'], inplace=True)
#df_salary_conversion['base_salary'].fillna(0, inplace=True)
df_it19_u['bonus'].fillna(0, inplace=True)
df_it19_u['stocks'].fillna(0, inplace=True)

df_it19_u['base_salary'].apply(pd.to_numeric, errors='coerce')
df_it19_u['bonus'].apply(pd.to_numeric, errors='coerce')
df_it19_u['stocks'].apply(pd.to_numeric, errors='coerce')
df_it19_u[['base_salary','bonus','stocks']] = df_it19_u[['base_salary','bonus','stocks']].astype('int64')

# Create a new 'salary' column by adding the three columns together
df_it19_u['salary_eur'] = df_it19_u['base_salary'] + df_it19_u['bonus'] + df_it19_u['stocks']
df_it19_u['salary'] = df_it19_u['salary_eur'] * eur2usd_2019

len_it19_salarydrop2 = len(df_it19_u) # For data quality report

df_it19_u.head(2)


# In[60]:


df_it19_u[['base_salary','bonus','stocks','salary']].info()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2020 </h3>

# In[62]:


top_5_values_per_column(df_it20_ini)


# <h4 style="background-color: #0AB89E; color: white; font-size: 20px; font-weight: bold; padding: 5px;">Renaming the columns</h4>

# In[64]:


# Adding year and country
df_it20_ini['year'] = 2020
df_it20_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position_': 'job_title',
    'total_years_of_experience': 'experience',
    'your_main_technology_/_programming_language': 'skills',
    'other_technologies/programming_languages_you_use_often': 'skills_2',
    'yearly_brutto_salary_(without_bonus_and_stocks)_in_eur': 'base_salary',
    'yearly_bonus_+_stocks_in_eur': 'bonus',
    'annual_brutto_salary_(without_bonus_and_stocks)_one_year_ago._only_answer_if_staying_in_the_same_country': 'salary_1y_ago',
    'annual_bonus+stocks_one_year_ago._only_answer_if_staying_in_same_country': 'bonus_1y_ago',
    'yearly_stocks_one_year_ago._only_answer_if_staying_in_same_country':'stocks_1y_ago',
    'main_language_at_work': 'language_at_work'
}

# Rename columns using the mappings
df_it20_u = df_it20_ini.rename(columns=column_mappings)
df_it20_u.head(2)


# <h4 style="background-color: #0AB89E; color: white; font-size: 20px; font-weight: bold; padding: 5px;">Checking cleanliness</h4>

# ##### TODO: clean_salary function should try to convert to float after checked the strings, and removed whitespaces and commas.

# In[67]:


# Cleaning report

df_it20_u['base_salary'] = df_it20_u['base_salary'].apply(clean_salary_0616, df_report=df_report, prefix='df_it20_u_salary')
df_it20_u['bonus'] = df_it20_u['bonus'].apply(clean_salary_0616, df_report=df_report, prefix='df_it20_u_bonus')
df_report


# <h4 style="background-color: #0AB89E; color: white; font-size: 20px; font-weight: bold; padding: 5px;">Converting to USD </h4>

# In[69]:


len_it20_salarydrop1 = len(df_it20_u)

df_it20_u = df_it20_u.dropna(subset=['base_salary'])
df_it20_u['bonus'].fillna(0, inplace=True)

#df_it20_u['bonus'] = df_it20_u['bonus'].str.replace('$', '')
#df_it20_u['bonus'] = df_it20_u['bonus'].str.replace('> ', '')

df_it20_u['base_salary'] = pd.to_numeric(df_it20_u['base_salary'], errors='coerce')
df_it20_u['bonus'] = pd.to_numeric(df_it20_u['bonus'], errors='coerce')

df_it20_u['base_salary'].fillna(0, inplace=True)
df_it20_u['bonus'].fillna(0, inplace=True)

#df_it20_u['base_salary'] = df_it20_u['base_salary'].astype('int64')
#df_it20_u['bonus'] = df_it20_u['bonus'].astype('int64')

# Create a new 'salary' column by adding the columns together
df_it20_u['salary_eur'] = df_it20_u['base_salary'] + df_it20_u['bonus']
df_it20_u['salary'] = df_it20_u['salary_eur'] * eur2usd_2020

len_it20_salarydrop2 = len(df_it20_u)

df_it20_u.head(2)


# In[70]:


df_it20_u[['base_salary','bonus','salary']].info()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2021 </h3>

# In[72]:


top_5_values_per_column(df_it21_ini)


# In[73]:


# In some cases the answer were given only in 'annual_brutto_salary_without_bonus_and_stocks_in_eur', and nothing in 'annual_brutto_salary_with_bonus_and_stocks_in_eur'
# Therefore I take the larger of the two
df_it21_ini['salary_eur'] = df_it21_ini[['annual_brutto_salary_without_bonus_and_stocks_in_eur', 'annual_brutto_salary_with_bonus_and_stocks_in_eur']].max(axis=1)


# In[74]:


df_it21_ini['year'] = 2021
df_it21_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position': 'job_title',
    'your_position': 'job_title_2',
    'total_years_of_experience': 'experience',
    'main_technology_/_programming_language': 'skills',
    'other_technologies/programming_languages_you_use_often': 'skills_2',
    'annual_brutto_salary_without_bonus_and_stocks_in_eur': 'base_salary',
    #'annual_brutto_salary_with_bonus_and_stocks_in_eur': 'salary_eur',
    'annual_bonus+stocks_one_year_ago.': 'salary_w_bonus_1y_ago',
    'what_languages_do_you_speak_and_use_at_work?': 'language_at_work'
}

# Rename columns using the mappings
df_it21_u = df_it21_ini.rename(columns=column_mappings)
df_it21_u.head(2)


# #### checking cleanliness

# In[76]:


df_it21_u['salary_eur'] = df_it21_u['salary_eur'].apply(clean_salary_0616, df_report=df_report, prefix='df_it21_u_salary')
df_report


# In[77]:


len_it21_salarydrop1 = len(df_it21_u)

# dropping initial Nones, that wouldn't be converted with pd.to_numeric errors='raise'
df_it21_u = df_it21_u.dropna(subset=['salary_eur'])

df_it21_u['salary_eur'].apply(pd.to_numeric, errors='raise')

# dropping Nones that is the result of pd.to_numeric
df_it21_u = df_it21_u.dropna(subset=['salary_eur'])

df_it21_u['salary_eur'] = df_it21_u['salary_eur'].astype('int64') # float64 can contain Nones, which is annoying. The data should be convertible to int64 (The only exception is if the answer was given as a float.)

# Create a new 'salary' column by adding the columns together
df_it21_u['salary'] = df_it21_u['salary_eur'] * eur2usd_2021

len_it21_salarydrop2 = len(df_it21_u)

df_it21_u.head(2)


# In[78]:


df_it21_u[['salary_eur','salary']].info()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2022 </h3>

# In[80]:


top_5_values_per_column(df_it22_ini)


# In[81]:


# In some cases the answer were given only in 'annual_brutto_salary_without_bonus_and_stocks_in_eur', and nothing in 'annual_brutto_salary_with_bonus_and_stocks_in_eur'
# Therefore I take the larger of the two
df_it22_ini['salary_eur'] = df_it22_ini[['annual_gross_salary_without_bonus_and_stocks_in_eur', 'annual_brutto_salary_with_bonus_and_stocks_in_eur']].max(axis=1)


# In[82]:


df_it22_ini['year'] = 2022
df_it22_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position': 'job_title',
    'your_position': 'job_title_2',
    'total_years_of_experience': 'experience',
    'main_technology_/_programming_language': 'skills',
    'other_technologies/programming_languages_you_use_often':'skills_2',
    'your_main_technology':'skills_3',
    'annual_gross_salary_without_bonus_and_stocks_in_eur': 'base_salary',
    'annual_brutto_salary_with_bonus_and_stocks_in_eur': 'salary_w_bonus',
    'annual_bonus+stocks_one_year_ago.':'bonus_1y_ago',
    'what_languages_do_you_speak_and_use_at_work?': 'language_at_work'
}

# Rename columns using the mappings
df_it22_u = df_it22_ini.rename(columns=column_mappings)


# In[83]:


df_it22_u['salary_eur'] = df_it22_u['salary_eur'].apply(clean_salary_0616, df_report=df_report, prefix='df_it22_u_salary_eur')
df_report


# In[84]:


# 
# 
# df_it22_u = df_it22_u.dropna(subset=['salary_eur'])
# df_it22_u['salary_eur'].apply(pd.to_numeric, errors='coerce')
# 
# # Create a new 'salary' column by adding the columns together
# df_it22_u['salary'] = df_it22_u['salary_eur'] * eur2usd_2022
# df_it22_u.head(2)


# In[85]:


len_it22_salarydrop1 = len(df_it22_u)

# dropping initial Nones, that wouldn't be converted with pd.to_numeric errors='raise'
df_it22_u = df_it22_u.dropna(subset=['salary_eur'])
df_it22_u['salary_eur'].apply(pd.to_numeric, errors='raise')

# dropping Nones that is the result of pd.to_numeric
df_it22_u = df_it22_u.dropna(subset=['salary_eur'])

df_it22_u['salary_eur'] = df_it22_u['salary_eur'].astype('int64') # float64 can contain Nones, which is annoying. The data should be convertible to int64 (The only exception is if the answer was given as a float.)

# Create a new 'salary' column by adding the columns together
df_it22_u['salary'] = df_it22_u['salary_eur'] * eur2usd_2022
len_it22_salarydrop2 = len(df_it22_u)

df_it22_u.head(2)


# In[86]:


df_it22_u[['salary_eur','salary']].info()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2023 </h3>

# In[88]:


top_5_values_per_column(df_it23_ini)


# In[89]:


# In some cases the answer were given only in 'annual_brutto_salary_without_bonus_and_stocks_in_eur', and nothing in 'annual_brutto_salary_with_bonus_and_stocks_in_eur'
# Therefore I take the larger of the two
df_it23_ini['salary_eur'] = df_it23_ini[['annual_gross_salary_without_bonus_and_stocks_in_eur', 'annual_gross_salary_with_bonus_and_stocks_in_eur']].max(axis=1)


# In[90]:


df_it23_ini['year'] = 2023
df_it23_ini['country'] = "de"

# Define column name mappings
column_mappings = {
    'position': 'job_title',
    'your_position': 'job_title_2',
    'total_years_of_experience': 'experience',
    'main_technology_/_programming_language': 'skills',
    'other_technologies/programming_languages_you_use_often':'skills_2',
    'your_main_technology':'skills_3',
    'annual_gross_salary_without_bonus_and_stocks_in_eur': 'base_salary',
    'annual_gross_salary_with_bonus_and_stocks_in_eur': 'salary_w_bonus',
    'annual_gross_salary+bonus+stocks_one_year_ago.':'salary_w_bonus_1y_ago',
    'what_languages_do_you_speak_and_use_at_work?': 'language_at_work'
}

# Rename columns using the mappings
df_it23_u = df_it23_ini.rename(columns=column_mappings)


# In[91]:


# Cleaning & report
df_it23_u['salary_eur'] = df_it23_u['salary_eur'].apply(clean_salary_0616, df_report=df_report, prefix='df_it23_u_salary_eur')
df_report


# In[92]:


len_it23_salarydrop1 = len(df_it23_u)

# dropping initial Nones, that wouldn't be converted with pd.to_numeric errors='raise'
df_it23_u = df_it23_u.dropna(subset=['salary_eur'])

df_it23_u['salary_eur'].apply(pd.to_numeric, errors='raise')

# dropping Nones that is the result of pd.to_numeric
df_it23_u = df_it23_u.dropna(subset=['salary_eur'])

df_it23_u['salary_eur'] = df_it23_u['salary_eur'].astype('int64') # float64 can contain Nones, which is annoying. The data should be convertible to int64 (The only exception is if the answer was given as a float.)

# Create a new 'salary' column by adding the columns together
df_it23_u['salary'] = df_it23_u['salary_eur'] * eur2usd_2023

len_it23_salarydrop2 = len(df_it23_u)

df_it23_u.head(2)


# In[93]:


df_it23_u[['salary_eur','salary']].info()


# ## Kaggle 

# ### TODO: what happens with <1000000 'salary_range' answers?

# In[96]:


def kaggle_summary(df):
    # Create an empty DataFrame to store the result
    result_df = pd.DataFrame(columns=['column_name', 'second_row_values', 'top_5_values'])
    
    # Iterate through each column
    for column in df.columns:
        # Find the five most common values in the column
        top_5_values = df[column].value_counts().head(5).index.tolist()
        
        # Get the value of the second row for the current column
        second_row_value = df.iloc[0][column]
        
        # Create a DataFrame with the current column name, top 5 values, and second row value
        data_to_append = pd.DataFrame({'column_name': [column],
                                       'second_row_values': [second_row_value],
                                       'top_5_values': [top_5_values]
                                       })
        
        # Concatenate the new DataFrame with the result DataFrame
        result_df = pd.concat([result_df, data_to_append], ignore_index=True)
    
    return result_df


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2019 </h3>

# In[98]:


result_df = kaggle_summary(df_k19_ini)
result_df.to_csv('../results/df_k19_structure.txt', sep='\t', index=True)
result_df.head(10)


# In[99]:


# Select only the relevant columns
df_k19_u1 = df_k19_ini[['q1', 'q3', 'q4', 'q5', 'q5_other_text', 'q6', 'q10', 'q15',]]
# Drop the first row as it's only an elaboration on the questionnaire.
df_k19_u1 = df_k19_u1.drop(df_k19_u1.index[0])

# Define column name mappings
column_mappings = {
    'q1': 'age',
    'q3': 'country',
    'q4': 'education_level',
    'q5': 'job_title',
    'q5_other_text': 'job_title_2',
    'q6': 'company_size',
    'q10': 'salary_range',
    'q15': 'experience'
}

# Rename columns using the mappings
df_k19_u = df_k19_u1.rename(columns=column_mappings)
#df_k19_u = df_k19_u1.drop(df_k19_u1.index[0])
df_k19_u['year'] = 2019
df_k19_u.head(3)


# In[100]:


len_k19_salarydrop1 = len(df_k19_u)

df_k19_u['salary_range'] = df_k19_u['salary_range'].str.replace('$', '', regex=False)
df_k19_u['salary_range'] = df_k19_u['salary_range'].str.replace('> ', '', regex=False)

# Dropping rows where salary is Null
df_k19_u.dropna(subset=['salary_range'], inplace=True)

# Split the salary_range column on the dash ('-') and convert to numeric values
df_k19_u[['lower_salary', 'upper_salary']] = df_k19_u['salary_range'].str.split('-', expand=True)
df_k19_u['lower_salary'] = df_k19_u['lower_salary'].str.replace(',', '').astype(float)
df_k19_u['upper_salary'] = df_k19_u['upper_salary'].str.replace(',', '').astype(float)

df_k19_u['salary'] = df_k19_u[['lower_salary', 'upper_salary']].mean(axis=1)

len_k19_salarydrop2 = len(df_k19_u)
df_k19_u.head()


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2020 </h3>

# In[102]:


result_df = kaggle_summary(df_k20_ini)
result_df.to_csv('../results/df_k20_structure.txt', sep='\t', index=True)
result_df.head(10)


# In[103]:


# Select only the important columns
df_k20_u1 = df_k20_ini[['q1', 'q3', 'q4', 'q5', 'q6', 'q20', 'q24']]

# Drop the first row as it's only an elaboration on the questionnaire.
df_k20_u1 = df_k20_u1.drop(df_k20_u1.index[0])

# Define column name mappings
column_mappings = {
    'q1': 'age',
    'q3': 'country',
    'q4': 'education_level',
    'q5': 'job_title',
    'q6': 'experience',
    'q20': 'company_size',
    'q24': 'salary_range'
}

# Rename columns using the mappings
df_k20_u = df_k20_u1.rename(columns=column_mappings)
df_k20_u['year'] = 2020
df_k20_u.head(2)


# In[104]:


len_k20_salarydrop1 = len(df_k20_u)

df_k20_u['salary_range'] = df_k20_u['salary_range'].str.replace('$', '', regex=False)
df_k20_u['salary_range'] = df_k20_u['salary_range'].str.replace('> ', '', regex=False)

# Dropping rows where salary is Null
df_k20_u.dropna(subset=['salary_range'], inplace=True)

# Split the salary_range column on the dash ('-') and convert to numeric values
df_k20_u[['lower_salary', 'upper_salary']] = df_k20_u['salary_range'].str.split('-', expand=True)
df_k20_u['lower_salary'] = df_k20_u['lower_salary'].str.replace(',', '').astype(float)
df_k20_u['upper_salary'] = df_k20_u['upper_salary'].str.replace(',', '').astype(float)

df_k20_u['salary'] = df_k20_u[['lower_salary', 'upper_salary']].mean(axis=1)

len_k20_salarydrop2 = len(df_k20_u)
df_k20_u.head(2)


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2021 </h3>

# In[106]:


result_df = kaggle_summary(df_k21_ini)
result_df.to_csv('../results/df_k21_structure.txt', sep='\t', index=True)
result_df.head(10)


# In[107]:


# Select only the relevant columns
df_k21_u1 = df_k21_ini[['q1', 'q3', 'q4', 'q5', 'q6', 'q20', 'q21', 'q25']]

# Drop the first row as it's only an elaboration on the questionnaire.
df_k21_u1 = df_k21_u1.drop(df_k21_u1.index[0])

# Define column name mappings
column_mappings = {
    'q1': 'age',
    'q3': 'country',
    'q4': 'education_level',
    'q5': 'job_title',
    'q6': 'experience',
    'q20': 'industry',
    'q21': 'company_size',
    'q25': 'salary_range'
}

# Rename columns using the mappings
df_k21_u = df_k21_u1.rename(columns=column_mappings)
df_k21_u['year'] = 2021
df_k21_u.head(2)


# In[108]:


len_k21_salarydrop1 = len(df_k21_u)

df_k21_u['salary_range'] = df_k21_u['salary_range'].str.replace('$', '', regex=False)
df_k21_u['salary_range'] = df_k21_u['salary_range'].str.replace('>', '', regex=False)

# Dropping rows where salary is Null
df_k21_u.dropna(subset=['salary_range'], inplace=True)

# Split the salary_range column on the dash ('-') and convert to numeric values
df_k21_u[['lower_salary', 'upper_salary']] = df_k21_u['salary_range'].str.split('-', expand=True)
df_k21_u['lower_salary'] = df_k21_u['lower_salary'].str.replace(',', '').astype(float)
df_k21_u['upper_salary'] = df_k21_u['upper_salary'].str.replace(',', '').astype(float)

df_k21_u['salary'] = df_k21_u[['lower_salary', 'upper_salary']].mean(axis=1)

len_k21_salarydrop2 = len(df_k21_u)
df_k21_u.head(3)


# <h3 style="background-color: #047c98; color: white; font-size: 20px; font-weight: bold; padding: 10px;"> Year: 2022 </h3>

# In[110]:


result_df = kaggle_summary(df_k22_ini)
result_df.to_csv('../results/df_k22_structure.txt', sep='\t', index=True)
result_df.head(10)


# In[111]:


# Select only the relevant columns
df_k22_u1 = df_k22_ini[['q2', 'q4', 'q5', 'q8', 'q11', 'q23', 'q24', 'q25', 'q29']]

# Drop the first row as it's only an elaboration on the questionnaire
df_k22_u1 = df_k22_u1.drop(df_k22_u1.index[0])

# Define column name mappings
column_mappings = {
    'q2': 'age',
    'q4': 'country',
    'q5': 'are_you_student',
    'q8': 'education_level',
    'q11': 'experience',
    'q23': 'job_title',
    'q24': 'industry',
    'q25': 'company_size',
    'q29': 'salary_range'
}

# Rename columns using the mappings
df_k22_u = df_k22_u1.rename(columns=column_mappings)
df_k22_u['year'] = 2022
df_k22_u.head(3)


# In[112]:


len_k22_salarydrop1 = len(df_k22_u)

df_k22_u['salary_range'] = df_k22_u['salary_range'].str.replace('$', '', regex=False)
df_k22_u['salary_range'] = df_k22_u['salary_range'].str.replace('>', '', regex=False)

# Dropping rows where salary is Null
df_k22_u.dropna(subset=['salary_range'], inplace=True)

# Split the salary_range column on the dash ('-') and convert to numeric values
df_k22_u[['lower_salary', 'upper_salary']] = df_k22_u['salary_range'].str.split('-', expand=True)
df_k22_u['lower_salary'] = df_k22_u['lower_salary'].str.replace(',', '').astype(float)
df_k22_u['upper_salary'] = df_k22_u['upper_salary'].str.replace(',', '').astype(float)

df_k22_u['salary'] = df_k22_u[['lower_salary', 'upper_salary']].mean(axis=1)

len_k22_salarydrop2 = len(df_k22_u)
df_k22_u.head(3)


# ## Final checking uniformity

# In[114]:


df_ai_u['salary'].describe()


# In[115]:


df_it18_u['salary'].describe()


# In[116]:


df_it19_u['salary'].describe()


# In[117]:


df_it20_u['salary'].describe()


# In[118]:


df_it21_u['salary'].describe()


# In[119]:


df_it22_u['salary'].describe()


# In[120]:


df_it23_u['salary'].describe()


# In[121]:


df_k19_u['salary'].describe()


# In[122]:


df_k20_u['salary'].describe()


# In[123]:


df_k21_u['salary'].describe()


# In[124]:


df_k22_u['salary'].describe()


# ## Final cleaning report

# In[126]:


df_report_t = df_report.T
df_report_t.rename(columns={0: 'Occurrence'}, inplace=True)
df_report_t.to_csv('../results/Cleaning_report.txt', sep='\t', index=True)


# In[127]:


len_it18_u = len(df_it18_u)
len_it19_u = len(df_it19_u)
len_it20_u = len(df_it20_u)
len_it21_u = len(df_it21_u)
len_it22_u = len(df_it22_u)
len_it23_u = len(df_it23_u)

len_k19_u = len(df_k19_u)
len_k20_u = len(df_k20_u)
len_k21_u = len(df_k21_u)
len_k22_u = len(df_k22_u)

len_ai_u = len(df_ai_u)


# In[128]:


print(len_it18_u / len_it18_ini),
print(len_it19_u / len_it19_ini),
print(len_it20_u / len_it20_ini),
print(len_it21_u / len_it21_ini),
print(len_it22_u / len_it22_ini),
print(len_it23_u / len_it23_ini),
print(len_k19_u  / len_k19_ini ),
print(len_k20_u  / len_k20_ini ),
print(len_k21_u  / len_k21_ini ),
print(len_k22_u  / len_k22_ini ),
print(len_ai_u   / len_ai_ini  )


# # Union of the yearly dataframes

# The survey data have been cleaned and been prepared for unionizing.\
# These dataframes may contain many interesting and nuanced questions, but for this project, only the common questions will be kept and unioned.\
# Therefore some columns will need to be dropped.

# In[131]:


top_5_values_per_column(df_it23_u)


# ## Germany IT Survey

# In[133]:


df_it18_u = df_it18_u[['age', 'city', 'job_title', 'seniority_level', 'language_at_work', 'company_size', 'company_type', 'salary', 'year', 'country']]

df_it19_u = df_it19_u[['age', 'city', 'job_title', 'seniority_level', 'experience', 'language_at_work', 'company_size', 'company_type', 'salary', 'year', 'country'
                      , 'skills']]

df_it20_u = df_it20_u[['age', 'city', 'job_title', 'seniority_level', 'experience', 'language_at_work', 'company_size', 'company_type', 'salary', 'year', 'country'
                      , 'skills', 'skills_2', 'employment_status', 'years_of_experience_in_germany']]

df_it21_u = df_it21_u[['city', 'job_title', 'job_title_2', 'seniority_level', 'experience', 'language_at_work', 'company_size', 'salary', 'year', 'country'
                      , 'skills', 'skills_2', 'employment_status', 'years_of_experience_in_germany', 'your_seniority_level']]
                       
df_it22_u = df_it22_u[['city', 'job_title', 'job_title_2', 'seniority_level', 'experience', 'language_at_work', 'company_size', 'salary', 'year', 'country'
                      , 'skills', 'skills_2', 'skills_3', 'employment_status', 'years_of_experience_in_germany', 'your_seniority_level']]                       

df_it23_u = df_it23_u[['city', 'job_title', 'job_title_2', 'seniority_level', 'experience', 'language_at_work', 'company_size', 'salary', 'year', 'country'
                      , 'skills', 'skills_2', 'skills_3', 'employment_status', 'years_of_experience_in_germany', 'your_seniority_level', 'company_industry']]                       


# In[134]:


df_it_uni = pd.concat([df_it18_u, df_it19_u, df_it20_u, df_it21_u, df_it22_u, df_it23_u])  #concat creates a copy
df_it_uni['survey'] = 'it'
df_it_uni = df_it_uni.reset_index(drop=True)
len_it_uni = len(df_it_uni)


# In[135]:


df_it_uni.head(2)


# ## Kaggle

# In[137]:


df_k_uni = pd.concat([df_k19_u, df_k20_u, df_k21_u, df_k22_u]) #concat creates a copy
df_k_uni['survey'] = 'k'
len_k_uni = len(df_k_uni)


# In[138]:


df_k_uni.head(2)


# ## AI-Jobs.net

# In[140]:


df_ai_uni = df_ai_u.copy() #For naming convention's sake, I create a copy, since for the previous surveys I needed a concat method, which also created copies.
df_ai_uni['survey'] = 'ai'
len_ai_uni = len(df_ai_uni)


# # Transformations after union

# ## Dropping values based on project scope

# Uknown salary values were dropped already. \
# But furthermore, we need to know the **Country, Seniority, Job title**. If any of those is missing, I'll drop the row from further investigation.

# ### Employment status

# Dropping Part-Time employees, freelancers, trainees, students.

# #### AI-Jobs.net

# In[147]:


df_ai_uni['employment_status'].unique()


# In[148]:


len_ai_employmentdrop1 = len(df_ai_uni)
df_ai_uni = df_ai_uni[df_ai_uni['employment_status'] == 'ft']
len_ai_employmentdrop2 = len(df_ai_uni)


# There are no students in this dataset, therefore I set the dropped student counter to 0.

# In[150]:


len_ai_studentdrop1 = 0
len_ai_studentdrop2 = 0


# #### DE IT-Survey

# ##### Dropping Nulls

# In[153]:


df_it_uni['employment_status'][df_it_uni['year'] == 2020].unique()


# In[154]:


len_it_employmentdrop1 = len(df_it_uni)

# Define the years to filter
years_to_filter = [2020, 2021, 2022, 2023]

# Separate the rows where the year is in the specified range
df_filtered_years = df_it_uni[df_it_uni['year'].isin(years_to_filter)]

# Drop rows with NaN employment_status in the filtered subset
df_filtered_years = df_filtered_years.dropna(subset=['employment_status'])

# Combine back the filtered rows with the rest of the DataFrame
df_it_uni = pd.concat([df_filtered_years, df_it_uni[~df_it_uni['year'].isin(years_to_filter)]], ignore_index=True)

len_it_employmentdrop2 = len(df_it_uni)


# In[155]:


df_it_uni['employment_status'][df_it_uni['year'] == 2020].unique()


# ##### Dropping freelance, parttime, student

# In[157]:


df_it_uni['employment_status'].unique()


# In[158]:


df_it_uni = df_it_uni.reset_index(drop=True)


# In[159]:


len_it_studentdrop1 = len(df_it_uni)
df_it_uni = df_it_uni[df_it_uni['employment_status'].isin(['full-time employee', 'founder', 'full-time position, part-time position, & self-employed (freelancing, tutoring)', 'full/part-time employee']) | df_it_uni['employment_status'].isna()]
len_it_studentdrop2 = len(df_it_uni)


# In[160]:


df_it_uni['employment_status'].unique()


# #### Kaggle

# In[162]:


df_k_uni['are_you_student'].unique()


# In[163]:


#df_k_uni[ (df_k_uni['are_you_student'].notna()) & (df_k_uni['are_you_student'] != 'no') ].head()


# In[164]:


# This drops row number 70k --> 10k !
# df_k_uni = df_k_uni[(df_k_uni['are_you_student'] == 'no')]

# By manual inspection it seems that many people left it unanswered. It's better to just filter out the explicit 'yes'.


# In[165]:


df_k_uni[df_k_uni['are_you_student'] == 'yes'].head()


# The dedicated 'are_you_student' column is **not** filled properly, therefore I omit this counter.\
# Furthermore, there is no dedicated employment status category, therefore I also omit this counter.

# In[167]:


#len_k_studentdrop1 = 
#df_k_uni = df_k_uni[df_k_uni['are_you_student'].isin(['yes'])]
#len_k_studentdrop2 = 


# ##### Dropping from 'experience' column

# In[169]:


df_k_uni['experience'].unique()


# In[170]:


len_k_noncoderdrop1 = len(df_k_uni)
df_k_uni = df_k_uni[df_k_uni['experience'] != 'i have never written code']
len_k_noncoderdrop2 = len(df_k_uni)


# ### Country

# #### AI-Jobs

# In[173]:


df_ai_uni['country'].sort_values().unique()


# In[174]:


len_ai_countrydrop1 = len(df_ai_uni)
df_ai_uni.dropna(subset=['country'], inplace=True)
len_ai_countrydrop2 = len(df_ai_uni)


# #### Kaggle

# In[176]:


df_k_uni['country'].sort_values().unique()


# In[177]:


len_k_countrydrop1 = len(df_k_uni)
df_k_uni = df_k_uni[df_k_uni['country'] != 'i do not wish to disclose my location']
df_k_uni = df_k_uni[df_k_uni['country'] != 'other']
len_k_countrydrop2 = len(df_k_uni)


# #### Germany IT-Survey

# This is a germany-specific survey, therefore I just set the counter to 0.

# In[180]:


len_it_countrydrop1 = 0
len_it_countrydrop2 = 0


# ### Seniority_level

# #### Ai-Jobs

# In[183]:


len_ai_senioritydrop1 = len(df_ai_uni)
df_ai_uni.dropna(subset=['seniority_level'], inplace=True)
len_ai_senioritydrop2 = len(df_ai_uni)


# #### Germany IT-Survey

# In[185]:


len_it_senioritydrop1 = len(df_it_uni)
df_it_uni.dropna(subset=['seniority_level'], inplace=True)
len_it_senioritydrop2 = len(df_it_uni)


# #### Kaggle

# I'll later transform 'experience' into seniority, therefore, for the counter I add this to the senioritydrop

# In[188]:


len_k_senioritydrop1 = len(df_k_uni)
df_k_uni.dropna(subset=['experience'], inplace=True)
len_k_senioritydrop2 = len(df_k_uni)


# ### Job-title

# #### AI-jobs.net

# In[191]:


len_ai_jobtitledrop1 = len(df_ai_uni)
df_ai_uni.dropna(subset=['job_title'], inplace=True)
len_ai_jobtitledrop2 = len(df_ai_uni)


# #### Germany IT-Survey

# In[193]:


len_it_jobtitledrop1 = len(df_it_uni)
df_it_uni.dropna(subset=['job_title'], inplace=True)
len_it_jobtitledrop2 = len(df_it_uni)


# #### Kaggle

# In[195]:


len_k_jobtitledrop1 = len(df_k_uni)
df_k_uni.dropna(subset=['job_title'], inplace=True)
len_k_jobtitledrop2 = len(df_k_uni)


# ## Uniformization

# ### Country Codes

# #### Kaggle

# In[199]:


df_k_uni['country'].sort_values().unique()


# In[200]:


# Dictionary to map country names to 2-letter country codes
country_to_code = {
    'france': 'fr',
    'india': 'in',
    'indonesia': 'id',
    'united states of america': 'us',
    'australia': 'au',
    'mexico': 'mx',
    'germany': 'de',
    'turkey': 'tr',
    'netherlands': 'nl',
    'nigeria': 'ng',
    'canada': 'ca',
    'greece': 'gr',
    'belgium': 'be',
    'singapore': 'sg',
    'italy': 'it',
    'ireland': 'ie',
    'taiwan': 'tw',
    'russia': 'ru',
    'brazil': 'br',
    'south africa': 'za',
    'poland': 'pl',
    'iran, islamic republic of...': 'ir',
    'ukraine': 'ua',
    'pakistan': 'pk',
    'chile': 'cl',
    'japan': 'jp',
    'egypt': 'eg',
    'south korea': 'kr',
    'belarus': 'by',
    'viet nam': 'vn',
    'colombia': 'co',
    'israel': 'il',
    'china': 'cn',
    'united kingdom of great britain and northern ireland': 'gb',
    'sweden': 'se',
    'bangladesh': 'bd',
    'portugal': 'pt',
    'tunisia': 'tn',
    'argentina': 'ar',
    'czech republic': 'cz',
    'spain': 'es',
    'hong kong (s.a.r.)': 'hk',
    'cameroon': 'cm',
    'saudi arabia': 'sa',
    'austria': 'at',
    'kenya': 'ke',
    'morocco': 'ma',
    'romania': 'ro',
    'hungary': 'hu',
    'republic of korea': 'kr',
    'norway': 'no',
    'ethiopia': 'et',
    'philippines': 'ph',
    'thailand': 'th',
    'denmark': 'dk',
    'switzerland': 'ch',
    'peru': 'pe',
    'sri lanka': 'lk',
    'ghana': 'gh',
    'malaysia': 'my',
    'united arab emirates': 'ae',
    'nepal': 'np',
    'iraq': 'iq',
    'new zealand': 'nz',
    'algeria': 'dz',
    'ecuador': 'ec',
    'uganda': 'ug',
    'kazakhstan': 'kz',
    'zimbabwe': 'zw',
    'latvia': 'lv'
}


# In[201]:


# Transform country names to 2-letter country codes
df_k_uni['country'] = df_k_uni['country'].map(country_to_code)
df_k_uni['country'].unique()


# ### Seniority level

# #### De-IT

# In[204]:


df_it_uni['seniority_level'].unique()


# In[205]:


# Define the mapping dictionary
seniority_mapping_it = {
    'head / principal': 'executive',
    'lead / staff': 'executive',
    'c-level executive manager': 'executive',
    'head': 'executive',
    'lead': 'executive',
    'director': 'executive',
    'manager': 'executive',
    'vp': 'executive',
    'c-level executive manager':'executive',
    'cto': 'executive',  # direCTOr
    'principal': 'executive',
    'c-level': 'executive',
    'middle': 'medior',
    'entry level': 'junior',
    'intern': 'other',
    'working student': 'other',
    'student': 'other',
    '800': 'other',
    'key': 'other',
    'no idea, there are no ranges in the firm ': 'other',
    'self employed': 'other',
    'work center executive': 'other',
    'no level ': 'other',
    'no level': 'other',
    'work center manager':'other'
}


# In[206]:


# Replace the seniority levels using the mapping dictionary
df_it_uni['seniority_level'] = df_it_uni['seniority_level'].replace(seniority_mapping_it)
df_it_uni['seniority_level'].unique()


# #### AI-Jobs.net

# In[208]:


df_ai_uni['seniority_level'].unique()


# In[209]:


# Define the mapping dictionary
seniority_mapping_ai = {
    'mi':'medior',
    'en': 'junior',
    'se': 'senior',
    'ex': 'executive'
}


# In[210]:


# Replace the seniority levels using the mapping dictionary
df_ai_uni['seniority_level'] = df_ai_uni['seniority_level'].replace(seniority_mapping_ai)
df_ai_uni['seniority_level'].unique()


# #### Kaggle

# In[212]:


df_k_uni['experience'].unique()


# In[213]:


# Mapping of experience intervals to seniority levels
experience_to_seniority = {
    '< 1 years': 'junior',
    '1-2 years': 'junior',
    '1-3 years': 'junior',
    '3-5 years': 'medior',
    '5-10 years': 'senior',
    '10-20 years': 'senior',
    '20+ years': 'executive'
}


# In[214]:


# Create the seniority_level column
df_k_uni['seniority_level'] = df_k_uni['experience'].map(experience_to_seniority)
df_k_uni.head()


# ## Additonal cleaning

# ### Germany-IT

# In[217]:


df_it_uni.head(1)


# #### 'experience' and 'years_of_experience_in_germany'

# 'experience' and 'years_of_experience_in_germany' columns are filled with unclean answers

# In[220]:


df_it_uni['experience'].unique()


# In[221]:


df_it_uni['years_of_experience_in_germany'].unique()


# In[222]:


df_report_additional = pd.DataFrame(index=[0])


# In[223]:


df_it_uni['experience'] = df_it_uni['experience'].apply(clean_salary_0616, df_report=df_report_additional, prefix='experience')
df_report_additional


# In[224]:


df_it_uni['experience'] = df_it_uni['experience'].apply(pd.to_numeric, errors='coerce')
df_it_uni['experience'] = df_it_uni['experience'][(df_it_uni['experience'] <= 100)]


# In[225]:


df_it_uni['years_of_experience_in_germany'] = df_it_uni['years_of_experience_in_germany'].apply(clean_salary_0616, df_report=df_report_additional, prefix='experience_in_de')
df_report_additional


# In[226]:


df_it_uni['years_of_experience_in_germany'] = df_it_uni['years_of_experience_in_germany'].apply(pd.to_numeric, errors='coerce')
df_it_uni['years_of_experience_in_germany'] = df_it_uni['years_of_experience_in_germany'][(df_it_uni['years_of_experience_in_germany'] <= 100)]


# In[227]:


df_it_uni['experience'].describe()


# #### City

# In[229]:


df_it_uni.groupby('city')['salary'].count().sort_values(ascending=False).head(6)


# In[230]:


df_it_uni.loc[df_it_uni['city'] == 'münchen', 'city'] = 'munich'


# In[231]:


df_it_uni.groupby('city')['salary'].count().sort_values(ascending=False).head(6)


# In[232]:


# Define the major cities to keep
major_cities = ['berlin', 'munich', 'frankfurt', 'hamburg', 'stuttgart']

# Create the 'city_category' column
df_it_uni['city_category'] = df_it_uni['city'].apply(lambda x: x if x in major_cities else 'other')


# #### Language at work

# In[234]:


df_it_uni['language_at_work'].unique()


# In[235]:


def categorize_language(language_entry, categories):
    language_entry = str(language_entry).lower()  # Convert language_entry to lower case string
    for category, keywords in categories.items():
        for keyword in keywords:
            pattern = re.escape(keyword.lower())  # Create regex pattern for keyword
            if re.search(pattern, language_entry):
                return category
    return 'Only other languages'  # For entries that don't match any category


# In[236]:


language_categories = {
    'German-speaking': ['german', 'deutsch'],
    'English-speaking (but not german)': ['english']
}

df_it_uni['language_category'] = df_it_uni['language_at_work'].apply(lambda x: categorize_language(x, language_categories))


# In[237]:


df_it_uni['language_category'].unique()


# In[238]:


df_it_uni.groupby('language_category')['salary'].count().sort_values(ascending=False).head(6)


# #### company_size

# In[240]:


df_it_uni.groupby('company_size')['salary'].count().sort_values(ascending=False)


# In[241]:


df_it_uni['company_size'] = df_it_uni['company_size'].replace({'10-50': '11-50'})
df_it_uni['company_size'] = df_it_uni['company_size'].replace({'50-100': '51-100'})
df_it_uni['company_size'] = df_it_uni['company_size'].replace({'100-1000': '101-1000'})


# In[242]:


df_it_uni.groupby('company_size')['salary'].count().sort_values(ascending=False)


# In[243]:


def categorize_company_size(size):
    """
    Categorize company size into small (s), medium (m), or large (l).
    Handles NaN values and unknown categories.
    
    Parameters:
    size: Company size value (string or NaN)
    
    Returns:
    string: 's', 'm', 'l', or 'unknown'
    """
    # Handle NaN values first using pandas.isna()
    if pd.isna(size):
        return 'unknown'
        
    # Convert to string to handle any numeric inputs
    size = str(size).lower().strip()
    
    # Direct mapping for known categories
    if size in ['s', 'm', 'l']:
        return size
        
    # Large companies
    elif size in ['1000+', '1000-9,999 employees', '10,000 or more employees', 
                 '> 10,000 employees', '>1000', '1000 or more employees']:
        return 'l'
        
    # Medium companies
    elif size in ['101-1000', '250-999 employees', '50-249 employees', '51-100',
                 '100-999', '51-1000']:
        return 'm'
        
    # Small companies
    elif size in ['11-50', 'up to 10', '0-49 employees', '<50', '1-50',
                 '0-50', 'under 50 employees']:
        return 's'
        
    # Any other value is considered unknown
    return 'unknown'


# In[244]:


# Apply the function to both DataFrames
df_it_uni['company_size_category'] = df_it_uni['company_size'].apply(categorize_company_size)
df_it_uni['company_size_category'].unique()


# #### 'company_industry'

# In[246]:


df_it_uni.groupby('company_industry')['salary'].count().sort_values(ascending=False)


# In[247]:


# Define the function to categorize industries
def categorize_industry(industry):
    if industry in ['information services, it, software development, or other technology']:
        return 'information technology'
    elif industry in ['financial services', 'insurance']:
        return 'financial services'
    elif industry == 'retail and consumer services':
        return 'retail and consumer services'
    elif industry == 'manufacturing, transportation, or supply chain':
        return 'manufacturing, transportation, or supply chain'
    elif industry == 'healthcare':
        return 'healthcare'
    else:
        return 'other'

# Create the 'industry_category' column
df_it_uni['industry_category'] = df_it_uni['company_industry'].apply(categorize_industry)


# #### company_type

# In[249]:


len(df_it_uni['company_type'].unique())


# This is a free-string cell. I will not try to make sense of it; look inside, this task would be pointless.

# ### Kaggle

# In[252]:


df_k_uni.head(1)


# #### Education level

# In[254]:


df_k_uni['education_level'].unique()


# In[255]:


df_k_uni['education_level'] = df_k_uni['education_level'].str.replace('no formal education past high school','no degree')
df_k_uni['education_level'] = df_k_uni['education_level'].str.replace('some college/university study without earning a bachelor’s degree','no degree')
df_k_uni = df_k_uni[(df_k_uni['education_level'] == 'bachelor’s degree') | (df_k_uni['education_level'] == 'master’s degree') | (df_k_uni['education_level'] == 'doctoral degree') | (df_k_uni['education_level'] == 'no degree')]


# #### company_size

# In[257]:


df_k_uni.groupby('company_size')['salary'].count().sort_values(ascending=False)


# In[258]:


df_k_uni['company_size_category'] = df_k_uni['company_size'].apply(categorize_company_size)
df_k_uni['company_size_category'].unique()


# #### experience

# In[260]:


df_k_uni.groupby('experience')['salary'].count().sort_values(ascending=False)


# In[261]:


df_k_uni['experience'] = df_k_uni['experience'].replace({'1-2 years': '1-3 years'})


# #### industry

# In[263]:


df_k_uni.groupby('industry')['salary'].count().sort_values(ascending=False)


# ## AI-jobs

# ### Company size

# In[266]:


# Remove rows where all columns are NaN
df_ai_uni = df_ai_uni.dropna(how='all')


# In[267]:


df_ai_uni['company_size_category'] = df_ai_uni['company_size'].apply(categorize_company_size)
df_ai_uni['company_size_category'].unique()


# In[268]:


df_ai_uni[df_ai_uni['company_size'].isna()].head()


# # Deriving new variables

# ## Country-standardized salary

# In[271]:


# GDP per capita data (in USD)
gdp_per_capita = {
    'fr': 40886, 'in': 2016, 'au': 65099, 'us': 76329, 'nl': 57025, 
    'de': 48717, 'ie': 87947, 'ru': 15270, 'gr': 19829, 'ua': 4533, 
    'pk': 1491, 'jp': 40066, 'br': 8697, 'kr': 31961, 
    'by': 7888, 'ng': 2229, 'gb': 46125, 'se': 53755, 'mx': 10657, 
    'ca': 54917, 'pt': 23758, 'pl': 17939, 'id': 4289, 'it': 34776, 
    'cz': 23906, 'es': 31688, 'cl': 14938, 'hk': 46544, 'za': 6001, 
    'ar': 10461, 'tr': 10674, 'il': 44162, 'tw': 34166, 'eg': 3801, 
    'ma': 3585, 'hu': 18390, 'co': 6214, 'no': 89111, 'th': 7775, 
    'ch': 93259, 'vn': 3704, 'sg': 59806, 'bd': 1964, 'ir': 2273, 
    'pe': 7002, 'ke': 2066, 'ro': 15786, 'cn': 12710, 'be': 50126, 
    'at': 52084, 'dz': 4094, 'nz': 45380, 'tn': 3840, 'ph': 3593, 
    'my': 11109, 'dk': 61612, 'sa': 20619, 'ae': 43103, 'np': 1192, 
    'lk': 3841, 'gh': 2396, 'et': 936, 'iq': 4922, 'ec': 6245, 
    'kz': 10153, 'ug': 817, 'cm': 1500, 'zw': 1098, 
    'lv': 21779, 'ge': 4804, 'lt': 25064, 'fi': 53012, 'hr': 15647, 'om': 25056, 'ba': 7568, 'ee': 28247, 
    'mt': 34127, 'lb': 4136, 'si': 28439, 'mu': 10256, 'am': 7018, 'qa': 87661, 'ad': 41992, 'md': 5714,
    'uz': 2255, 'cf': 427, 'kw': 41079, 'cy': 32048, 'as': 19673, 'cr': 13365, 'pr': 35208, 'bo': 3600,
    'do': 8793, 'hn': 2736, 'bg': 12623, 'je': 55820, 'rs': 9260,  'lu': 125006
}


# In[272]:


country_salary_stats


# In[273]:


#df_ai_uni.drop(['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd','salary_normmed','salary_normmean','salary_normgdp','salary_normse'], axis=1, inplace=True)


# In[274]:


#df_k_uni.drop(['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd','salary_normmed','salary_normmean','salary_normgdp','salary_normse'], axis=1, inplace=True)


# In[275]:


#df_it_uni.drop(['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd','salary_normmed','salary_normmean','salary_normgdp','salary_normse'], axis=1, inplace=True)


# In[276]:


df_name = df_ai_uni

# Merge df with country_salary_stats to get the median income for each country
df_name = df_name.merge(
    country_salary_stats[['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd']],
    left_on='country',
    right_on='country_code',
    how='left'
)

# Calculate the normalized salaries
df_name['salary_normmed'] = df_name['salary'] / df_name['median_income_2020_usd']
df_name['salary_normmean'] = df_name['salary'] / df_name['mean_income_2020_usd']
df_name['salary_normgdp'] = df_name['salary'] / df_name['gdp_ppp_usd']
df_name['salary_normse'] = df_name['salary'] / df_name['glassdoor_software_engineer_usd']

df_ai_uni = df_name


# In[277]:


df_name = df_k_uni

# Merge df with country_salary_stats to get the median income for each country
df_name = df_name.merge(
    country_salary_stats[['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd']],
    left_on='country',
    right_on='country_code',
    how='left'
)

# Calculate the normalized salaries
df_name['salary_normmed'] = df_name['salary'] / df_name['median_income_2020_usd']
df_name['salary_normmean'] = df_name['salary'] / df_name['mean_income_2020_usd']
df_name['salary_normgdp'] = df_name['salary'] / df_name['gdp_ppp_usd']
df_name['salary_normse'] = df_name['salary'] / df_name['glassdoor_software_engineer_usd']

df_k_uni = df_name


# In[278]:


df_name = df_it_uni

# Merge df with country_salary_stats to get the median income for each country
df_name = df_name.merge(
    country_salary_stats[['country_code', 'median_income_2020_usd', 'mean_income_2020_usd', 'gdp_ppp_usd', 'glassdoor_software_engineer_usd']],
    left_on='country',
    right_on='country_code',
    how='left'
)

# Calculate the normalized salaries
df_name['salary_normmed'] = df_name['salary'] / df_name['median_income_2020_usd']
df_name['salary_normmean'] = df_name['salary'] / df_name['mean_income_2020_usd']
df_name['salary_normgdp'] = df_name['salary'] / df_name['gdp_ppp_usd']
df_name['salary_normse'] = df_name['salary'] / df_name['glassdoor_software_engineer_usd']

df_it_uni = df_name


# ### Approach 2

# In[280]:


df_ai_uni['country'][~(df_ai_uni['country'].isin(gdp_per_capita))].unique()


# In[281]:


# Normalize the salary - Kaggle
df_k_uni['salary_norm'] = df_k_uni.apply(lambda x: x['salary'] / gdp_per_capita[x['country']], axis=1)
df_k_uni.head(2)


# In[282]:


# Normalize the salary - AI-Jobs.net
df_ai_uni['salary_norm'] = df_ai_uni.apply(lambda x: x['salary'] / gdp_per_capita[x['country']], axis=1)
df_ai_uni.head()


# In[283]:


# Normalize the salary - Germany IT-Survey
df_it_uni['salary_norm'] = df_it_uni.apply(lambda x: x['salary'] / gdp_per_capita[x['country']], axis=1)
df_it_uni.head()


# ## Year-Standardized salary

# In[285]:


inflation_rates = {
    2018: 1.019,  # USD inflation from 2018 to 2019 (22.66% increase from 2018 to 2024)
    2019: 1.018,  # USD inflation from 2019 to 2020 (20.37% increase from 2019 to 2024)
    2020: 1.012,  # USD inflation from 2020 to 2021 (18.25% increase from 2020 to 2024)
    2021: 1.040,  # USD inflation from 2021 to 2022 (16.84% increase from 2021 to 2024)
    2022: 1.070,  # USD inflation from 2022 to 2023 (12.35% increase from 2022 to 2024)
    2023: 1.050,  # USD inflation from 2023 to 2024 (5.00% increase from 2023 to 2024)
    2024: 1.000   # base year, no inflation adjustment for 2024, 
}

# Calculate cumulative inflation adjustment factor
def calculate_cumulative_inflation(start_year, end_year=2024):
    if start_year >= end_year:
        return 1.0
    inflation_factors = [inflation_rates[year] for year in range(start_year, end_year)]
    cumulative_inflation = 1.0
    for factor in inflation_factors:
        cumulative_inflation *= factor
    return cumulative_inflation

# Apply inflation adjustment to the salary column
def salary_to_2024(row):
    return row['salary'] * calculate_cumulative_inflation(row['year'])

# Apply inflation adjustment to the salary column
def salarynorm_to_2024(row):
    return row['salary_norm'] * calculate_cumulative_inflation(row['year'])


# In[286]:


# Apply inflation adjustment directly to the 'salary' and 'salary_norm' columns
df_it_uni['salary_2024'] = df_it_uni.apply(lambda row: row['salary'] * calculate_cumulative_inflation(row['year']), axis=1)
df_it_uni['salary_norm_2024'] = df_it_uni.apply(lambda row: row['salary_norm'] * calculate_cumulative_inflation(row['year']), axis=1)
df_it_uni['salary_normmed_2024'] = df_it_uni.apply(lambda row: row['salary_normmed'] * calculate_cumulative_inflation(row['year']), axis=1)
df_it_uni['salary_normmean_2024'] = df_it_uni.apply(lambda row: row['salary_normmean'] * calculate_cumulative_inflation(row['year']), axis=1)
df_it_uni['salary_normgdp_2024'] = df_it_uni.apply(lambda row: row['salary_normgdp'] * calculate_cumulative_inflation(row['year']), axis=1)
df_it_uni['salary_normse_2024'] = df_it_uni.apply(lambda row: row['salary_normse'] * calculate_cumulative_inflation(row['year']), axis=1)

#df_it_uni['salary_2024'] = df_it_uni.apply(salary_to_2024, axis=1)
#df_it_uni['salary_norm_2024'] = df_it_uni.apply(salarynorm_to_2024, axis=1)
#df_it_uni['salary_normmed_2024'] = df_it_uni.apply(salarynorm_to_2024, axis=1)
#df_it_uni['salary_normmean_2024'] = df_it_uni.apply(salarynorm_to_2024, axis=1)
#df_it_uni['salary_normgdp_2024'] = df_it_uni.apply(salarynorm_to_2024, axis=1)
#df_it_uni['salary_normse_2024'] = df_it_uni.apply(salarynorm_to_2024, axis=1)

# Display the updated dataframe
df_it_uni[['year', 'salary', 'salary_2024', 'salary_norm', 'salary_norm_2024', 'salary_normmed_2024', 'salary_normmean_2024', 'salary_normgdp_2024', 'salary_normse_2024']].head()


# In[287]:


# Apply inflation adjustment directly to the 'salary' and 'salary_norm' columns
df_k_uni['salary_2024'] = df_k_uni.apply(lambda row: row['salary'] * calculate_cumulative_inflation(row['year']), axis=1)
df_k_uni['salary_norm_2024'] = df_k_uni.apply(lambda row: row['salary_norm'] * calculate_cumulative_inflation(row['year']), axis=1)
df_k_uni['salary_normmed_2024'] = df_k_uni.apply(lambda row: row['salary_normmed'] * calculate_cumulative_inflation(row['year']), axis=1)
df_k_uni['salary_normmean_2024'] = df_k_uni.apply(lambda row: row['salary_normmean'] * calculate_cumulative_inflation(row['year']), axis=1)
df_k_uni['salary_normgdp_2024'] = df_k_uni.apply(lambda row: row['salary_normgdp'] * calculate_cumulative_inflation(row['year']), axis=1)
df_k_uni['salary_normse_2024'] = df_k_uni.apply(lambda row: row['salary_normse'] * calculate_cumulative_inflation(row['year']), axis=1)

# Display the updated dataframe
df_k_uni[['year', 'salary', 'salary_2024', 'salary_norm', 'salary_norm_2024', 'salary_normmed_2024', 'salary_normmean_2024', 'salary_normgdp_2024', 'salary_normse_2024']].head()


# In[288]:


df_ai_uni.dropna(subset=['year'], inplace=True)


# In[289]:


df_ai_uni['year'] = df_ai_uni['year'].astype('int64')


# In[290]:


# Apply inflation adjustment directly to the 'salary' and 'salary_norm' columns
df_ai_uni['salary_2024'] = df_ai_uni.apply(lambda row: row['salary'] * calculate_cumulative_inflation(row['year']), axis=1)
df_ai_uni['salary_norm_2024'] = df_ai_uni.apply(lambda row: row['salary_norm'] * calculate_cumulative_inflation(row['year']), axis=1)
df_ai_uni['salary_normmed_2024'] = df_ai_uni.apply(lambda row: row['salary_normmed'] * calculate_cumulative_inflation(row['year']), axis=1)
df_ai_uni['salary_normmean_2024'] = df_ai_uni.apply(lambda row: row['salary_normmean'] * calculate_cumulative_inflation(row['year']), axis=1)
df_ai_uni['salary_normgdp_2024'] = df_ai_uni.apply(lambda row: row['salary_normgdp'] * calculate_cumulative_inflation(row['year']), axis=1)
df_ai_uni['salary_normse_2024'] = df_ai_uni.apply(lambda row: row['salary_normse'] * calculate_cumulative_inflation(row['year']), axis=1)

# Display the updated dataframe
df_ai_uni[['year', 'salary', 'salary_2024', 'salary_norm', 'salary_norm_2024', 'salary_normmed_2024', 'salary_normmean_2024', 'salary_normgdp_2024', 'salary_normse_2024']].head()


# ## Categorizing Job-titles

# In[292]:


#df_it['job_category_kw'] = df_it['job_title'].apply(categorize_by_keywords, category_keywords)
## df_it['job_category_kw'] = df_it['job_title'].apply(lambda x: categorize_by_keywords_1627(x, category_keywords))
#df_it['job_category'] = df_it['job_title'].apply(lambda x: categorize_job_title(x, categories))


# In[293]:


# Function to categorize job titles based on keyword match
def categorize_job_title_1945(job_title, categories):
    job_title = str(job_title).lower()  # Convert job_title to lower case string
    for category, keywords in categories.items():
        for keyword in keywords:
            pattern = re.escape(keyword.lower())  # Create regex pattern for keyword
            if re.search(pattern, job_title):
                return category
    return 'Uncategorized'  # For job titles that don't match any category


# In[294]:


job_categories = {
    'Project managers': [
        'project manager', 'pm' , 'program manager', 'project manager ', 'projectingenieur', 'project manager (pm)', 'program/project manager', 'technical lead', 
        'project manager & scrum master', 'technical project manager', 'digital project manager', 'it project manager', 
        'scrum master', 'scrum master / agile coach', 'sr project manager', 'senior project manager', 'senior program manager', 'engineering project manager', 
        'agile project manager', 'project leader', 'director of engineering', 'director of project management', 'director of technology', 'director of operations', 
        'project consultant', 'project coordinator', 'project supervisor', 'project assistant', 'project administrator', 'project management officer', 
        'program manager (technical)', 'construction project manager', 'it program manager', 'it project coordinator', 'it project management consultant', 
        'it project manager ', 'associate project manager', 'project portfolio manager', 'project office manager', 'product manager', 
        'technical program manager', 'digital transformation project manager', 'technical program manager (tpm)', 'digital project lead', 'delivery manager', 
        'global project manager', 'global program manager', 'business program manager', 'service delivery manager', 'it delivery manager', 'operations project manager', 
        'customer project manager', 'implementation project manager', 'senior delivery manager', 'business development manager operations', 'project & operations manager', 
        'it operations manager', 'manager (program, project, operations, executive-level, etc)', 'project & operations manager', 'technical project lead'
    ],
    'Team leaders': ['team lead', 'team leader'],
    'Leaders': ['head of', 'lead', 'principal', 'staff', 'vp', 'cto'],
    'Other managers': ['manager'],
    'Full Stack Developers': ['full stack', 'full-stack', 'fullstack'],
    'Architects': ['architect', 'data modeler', 'architekt'],
    'Cloud': ['cloud engineer', 'cloud consulting', 'cloud platform engineer', 'cloud infrastructure engineer', 'cloud automation engineer'],
    
    'PHP Developers': ['php'],
    'SAP Specialists': ['sap'],
    'NET Developers': ['.net', 'c#'],
    'C++ Developers': ['c++'],
    'Mobile': ['ios','mobile', 'android','application'],
    'Java/Scala Developers': ['java', 'scala','javascript','js', 'angular'],
    'Other languages': ['python', 'ruby', 'oracle', 'erlang', 'go', 'golang', 'pyhon'],
    
    'Embedded Engineers': ['embedded'],
    'Front End': ['front end', 'front-end','frontend','frontent'],
    'Back End': ['back end','back-end', 'backend'],
    'Web developer': ['web developer', 'webdev', 'web-entwickler'],
    'Game': ['unreal','game','unity','unity3d'],
    'Hardware':['hardware'],
    'Security':['security'],
    'Database Dev & Admin':['dba', 'database developer', 'database engineer', 'database administrator', 'database manager', 'databengineer', 'data administrator'],
    'System admin': ['sys admin', 'sysadmin', 'system administrator', 'systems administrator', 'it administration', 'it admin', 'network administrator'],
    'Statisticians': ['statistician'],
    'Consultant': ['consultant', 'berater', 'consulter', 'consulting'],
    
    'Researcher': ['researcher'],
    'Prompt Engineer': ['prompt'],
    'Bioinformatics': ['bioinformatics', 'biostatistics', 'computational biologist'],
    
    'Business Analyst': [
        'business analyst', 'business intelligence analyst', 'bi analyst', 'bi specialist',
        'business insights analyst', 'financial data analyst', 'compliance data analyst', 'product data analyst', 'marketing data analyst', 'business data analyst', 
        'data analyst (business, marketing, financial, quantitative, etc)', 'business intelligence manager', 'business intelligence engineer', 'business intelligence specialist', 
        'business intelligence data analyst', 'business analyst/re', 'business analyst ', 'business analyst / business development manager operations', 
        'business development manager operations'
    ],
    
    'Data Analyst': [
        'data analyst', 'business intelligence developer', 'bi developer', 'research analyst', 'analytics engineer', 'data management analyst', 'data visualization',
        'data strategist',
        'data analytics associate', 'product analyst', 'marketing analyst', 'dana analyst', 'data reporting analyst', 'data quality analyst', 'finance data analyst', 
        'compliance data analyst', 'product data analyst', 'marketing data analyst', 'financial data analyst', 'data analytics consultant', 'data integration analyst', 
        'insight analyst', 'data analyst (business, marketing, financial, quantitative, etc)', 'business data analyst', 'data modeller', 'data analytics manager', 
        'data operations analyst', 'data quality manager', 'data science analyst', 'data specialist', 'data strategy manager', 'data management consultant', 
        'data analytics lead', 'data analytics specialist', 'data operations manager', 'data product analyst', 'data product owner', 'data quality engineer',
        'data visualization analyst', 'data visualization specialist'
    ],
    
    'Business Analyst': [
        'business intelligence',
    ],
    
    'Data Engineer': [
        'ml ops engineer', 'data engineer', 'database engineer', 'big data engineer', 'etl developer', 'etl engineer', 'big data developer', 'data operations',
        'machine learning operations engineer','machine learning infrastructure engineer', 'machine learning developer',
        'data integration', 'data processing', 'data developer', 'data integration engineer', 'data pipeline engineer', 'cloud data engineer', 'data infrastructure engineer', 
        'data warehouse engineer', 'data migration engineer', 'etl/data engineer', 'data platform engineer', 'data ops engineer', 'data services engineer', 
        'data solutions engineer', 'data systems engineer', 'data automation engineer', 'data engineering manager', 'data engineer lead', 'data engineering consultant', 
        'data operations engineer', 'data modeling engineer', 'data engineering analyst', 'data warehouse developer', 'data engineer/scientist', 'data quality engineer', 
        'data mining engineer', 'data software engineer', 'data engineering specialist'
    ],
    
    'Data Scientist/ ML Engineer': [
        'ai developer', 'deep learning engineer', 'data science', 'data scientist', 'machine learning engineer', 'ai engineer', 'ml engineer', 'research scientist', 'deep learning engineer', 
        'machine learning specialist', 'ai programmer', 'ai scientist', 'decision scientist',
        'nlp engineer', 'computer vision engineer', 'applied scientist', 'ai/ml engineer', 'data scientist/analyst', 'data science engineer', 'data science analyst', 
        'data science manager', 'machine learning scientist', 'applied ml engineer', 'ai/ml scientist', 'research engineer', 'mlops engineer', 'data scientist lead', 
        'data scientist manager', 'senior data scientist', 'principal data scientist', 'staff machine learning engineer', 'staff data scientist', 'machine learning software engineer',
        'machine learning manager', 'machine learning ops engineer', 'principal machine learning engineer', 'applied machine learning engineer', 'ml engineer/analyst', 
        'data & applied scientist', 'machine learning research engineer', 'machine learning modeler', 'ai/ml researcher', 'mlops/data scientist', 'ai/ ml research engineer',
        'ml engineer/research scientist', 'ai/ml engineer/researcher', 'machine learning engineet'
    ],
    
    'Data Governance & Compliance': [
        'data governance specialist', 'data governance manager', 'compliance data analyst', 'data quality manager', 'data quality analyst', 'data quality engineer', 'data management analyst', 'data management consultant', 'data management specialist', 'data privacy officer', 'data protection officer', 'data security officer', 'data compliance manager', 'risk and compliance analyst', 'risk and compliance manager', 'compliance officer', 'data stewardship specialist', 'data steward', 'regulatory compliance analyst', 'regulatory compliance manager', 'information governance specialist', 'information governance manager', 'data governance consultant'
    ],
    
    'Software Engineer':[
        'software engineer', 'software developer'
    ],
    
    'DevOps Engineer': [
        'devops engineer', 'devops', 'devops engineer ',
        'software engineer (devops)', 'system engineer', 'system administrator', 'systems engineer', 
        'it infrastructure consultant', 'it consultant', 'solution engineer', 'lead devops', 'lead devops engineer', 'technical lead devops', 
        'sr. devops', 'sr. engineer', 'sre', 'sre engineer', 'site reliability engineer', 'site reliability engineer '
    ],
    
    'UI/UX Designers': [
        'ux designer', 'ui designer', 'ux/ui designer', 'designer (ui, ux)', 'designer (ui/ux)', 'product designer', 'interaction designer', 'user experience designer', 'visual designer', 'frontend designer', 'creative designer', 'web designer', 'graphic designer', 'mobile designer', 'user interface designer', 'design lead', 'design researcher', 'ui/ux specialist', 'ux/ui lead', 'digital designer', 'ui/ux consultant'
    ],
    
    'QA/Test Engineers': [
        'qa', 'testing', 'tester', 'test', 'qa test engineer', 'qa engineer', 'qa automation engineer', 'automation qa', 'test automation engineer', 'manual qa', 
        'qa automation', 'qa consultant', 'qa analyst', 'software test engineer', 'quality engineer', 'test engineer', 'test manager', 'testautomation', 
        'automation test engineer', 'quality assurance', 'quality assurance engineer', 'qa specialist', 'qa automation specialist', 'qa automation'
    ],
    
    'Other Engineers': [
        'platform engineer', 'engineer (non-software)', 'network engineer', 'support engineer', 'electrical engineer', 'firmware engineer', 'robotics engineer', 'it engineer', 'reporting engineer', 'ta engineer', 'cisco engineer'     
    ],
    
    'Other Developers': [
        'web developer', 'sw developer', 'softwaredeveloper', 'xr', 'crypto', 'rpa developer', 'sharepoint developer', 'nav developer', 'dwh developer', 'web deleloper', 'erp developer'
    ],
    
    'Out of scope': [
        'teacher', 'professor', 'lawyer', 'sales', 'pcb designer', 'coach', 'producer', 'recruiter', 'agile', 'banker', 'quant'
    ],
    'System...': [
        'system'
    ],
    'Advocacy':['developer advocate', 'developer relations/advocacy', ],
    
    'Too vague answers': [
        'engineer', 'developer', 'designer', 'support', 'operations', 'analyst', 'spezialist', 'specialist'
    ],
    '"Other"': ['other']
    
}


# ### DE IT-Survey

# In[296]:


df_it_uni['job_title'] = df_it_uni['job_title'].str.replace('senior', '', case=False, regex=False)
df_it_uni['job_title'] = df_it_uni['job_title'].str.replace('sr.', '', case=False, regex=False)
df_it_uni['job_title'] = df_it_uni['job_title'].str.strip()


# In[297]:


df_it_uni['job_category'] = df_it_uni['job_title'].apply(lambda x: categorize_job_title_1945(x, job_categories))


# In[298]:


# Plot histogram for the 'job_category' column
plt.figure(figsize=(8, 3))
df_it_uni['job_category'].value_counts().plot(kind='bar')
plt.title('Histogram of Job Categories')
plt.xlabel('Job Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[299]:


df_it_uni['job_title'][df_it_uni['job_category'] == 'Uncategorized'].tail(50)
#df_it_uni['job_title'][df_it_uni['job_category'] == 'Too vague answers'].tail(50)


# In[300]:


df_it_uni['job_title'][df_it_uni['job_category'] == 'Database Development/ Admininistration'].head(50)


# ### AI-Jobs.net

# In[302]:


df_ai_uni.loc['job_title'] = df_ai_uni['job_title'].str.replace('senior', '', case=False, regex=False)
df_ai_uni.loc['job_title'] = df_ai_uni['job_title'].str.replace('sr.', '', case=False, regex=False)
df_ai_uni.loc['job_title'] = df_ai_uni['job_title'].str.strip()


# In[303]:


df_ai_uni['job_category'] = df_ai_uni['job_title'].apply(lambda x: categorize_job_title_1945(x, job_categories))


# In[304]:


# Plot histogram for the 'job_category' column
plt.figure(figsize=(8, 3))
df_ai_uni['job_category'].value_counts().plot(kind='bar')
plt.title('Histogram of Job Categories')
plt.xlabel('Job Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[305]:


df_ai_uni['job_title'][df_ai_uni['job_category'] == 'Uncategorized'].head(50)
#df_ai_uni['job_title'][df_ai_uni['job_category'] == 'Too vague answers'].tail(50)


# ### Kaggle

# In[307]:


df_k_uni['job_title'] = df_k_uni['job_title'].str.replace('senior', '', case=False, regex=False)
df_k_uni['job_title'] = df_k_uni['job_title'].str.replace('sr.', '', case=False, regex=False)
df_k_uni['job_title'] = df_k_uni['job_title'].str.strip()


# In[308]:


df_k_uni['job_category'] = df_k_uni['job_title'].apply(lambda x: categorize_job_title_1945(x, job_categories))


# In[309]:


# Plot histogram for the 'job_category' column
plt.figure(figsize=(8, 3))
df_k_uni['job_category'].value_counts().plot(kind='bar')
plt.title('Histogram of Job Categories')
plt.xlabel('Job Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# In[310]:


df_k_uni['job_title'][df_k_uni['job_category'] == 'Uncategorized'].tail(50)
#df_k_uni['job_title'][df_k_uni['job_category'] == 'Too vague answers'].head(50)


# In[311]:


for category in job_categories:
    unique_titles = df_it_uni['job_title'][df_it_uni['job_category'] == category].unique()
    print(f"{category}: {unique_titles}")


# In[312]:


for category in job_categories:
    unique_titles = df_ai_uni['job_title'][df_ai_uni['job_category'] == category].unique()
    print(f"{category}: {unique_titles}")


# In[313]:


for category in job_categories:
    unique_titles = df_k_uni['job_title'][df_k_uni['job_category'] == category].unique()
    print(f"{category}: {unique_titles}")


# ## Western Countries

# In[315]:


western_countries = [
    'al', 'ad', 'am', 'at', 'az', 'by', 'be', 'ba', 'bg', 'hr',
    'cy', 'cz', 'dk', 'ee', 'fi', 'fr', 'ge', 'de', 'gr', 'hu',
    'is', 'ie', 'it', 'kz', 'xk', 'lv', 'li', 'lt', 'lu', 'mt',
    'md', 'mc', 'me', 'nl', 'mk', 'no', 'pl', 'pt', 'ro', 'ru',
    'sm', 'rs', 'sk', 'si', 'es', 'se', 'ch', 'tr', 'ua', 'gb',
    'va', 'ca', 'au', 'us'
]


# In[316]:


developed_countries = ['de', 'gb', 'nl', 'se', 'dk', 'be', 'fi', 'at', 'ch', 'ie', 'ca', 'au', 'us']


# ## Log-transformed salary

# The reasonability of this step was assessed later, in the Analysis part, and iteratively added back to here.

# In[319]:


dataframes = [df_k_uni, df_it_uni, df_ai_uni]
columns_to_transform = ['salary', 'salary_2024', 'salary_norm', 'salary_norm_2024', 'salary_normmed_2024', 'salary_normmean_2024', 'salary_normgdp_2024', 'salary_normse_2024']

for df in dataframes:
    for col in columns_to_transform:
        df[f'{col}_log'] = np.log(df[col])


# # Outlier detection

# In[321]:


def add_z_scores(data, numerical_column):
    # Calculate Z-scores
    mean = data[numerical_column].mean()
    std = data[numerical_column].std()
    z_scores = (data[numerical_column] - mean) / std
    data['z_score'] = z_scores
    
    # Calculate modified Z-scores
    # 0.6745 * (xi - median)/MAD, where xi is the actual row that the Z-score will be calculated to. MAD is the median absolute deviation.
    median = data[numerical_column].median()
    mad = np.median(np.abs(data[numerical_column] - median))
    modif_z_scores = 0.6745 * np.abs(data[numerical_column] - median) / mad
    data['modif_z_score'] = modif_z_scores

    modif_z_scores_signed = 0.6745 * (data[numerical_column] - median) / mad
    data['modif_z_score_signed'] = modif_z_scores_signed
    
    return data


# In[322]:


def detect_outliers(data):
    threshold = 10  # Adjust threshold as needed
    median = data.median()
    median_absolute_deviation = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * np.abs(data - median) / median_absolute_deviation
    return modified_z_scores > threshold


# In[323]:


def detect_outliers_1006_2035(data, numerical_column, lower_threshold=-3, upper_threshold=10):
    median = data[numerical_column].median()
    mad = np.median(np.abs(data[numerical_column] - median))
    modif_z_scores_signed = 0.6745 * (data[numerical_column] - median) / mad
    outliers = data[(modif_z_scores_signed < lower_threshold) | (modif_z_scores_signed > upper_threshold)]
    return outliers


# ## Based on Normalized, Log-transformed salary

# ### Kaggle

# In[326]:


df_k_uni_copy = df_k_uni.copy()
df_ai_uni_copy = df_ai_uni.copy()
df_it_uni_copy = df_it_uni.copy()


# In[327]:


# grouped DF
df_k_g = df_k_uni.groupby(['experience'])

# adding Z-scores
df_k_gz = df_k_g.apply(lambda x: add_z_scores(x, 'salary_normmed_2024_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_k_gz.sort_values('modif_z_score_signed', ascending=True).head(5)


# In[328]:


# Sorting the dataframes for plotting
df_k_gzs = df_k_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_k_gmzs = df_k_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_k_gmzs_signed = df_k_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_k_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_k_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_k_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# In[329]:


# Start with the initial DataFrame
len_k_initial = len(df_k_gz)

# Define thresholds
lower_threshold = -0.5
upper_threshold = 3.0

# Boolean masks for outliers
mask_k_small = df_k_gz['modif_z_score_signed'] < lower_threshold
mask_k_large = df_k_gz['modif_z_score_signed'] > upper_threshold

# Count outliers
len_k_outlierdrop_small = mask_k_small.sum()
len_k_outlierdrop_large = mask_k_large.sum()

# Remove outliers
df_k = df_k_gz[~(mask_k_small | mask_k_large)].copy()

# Print results
print(f"{len_k_outlierdrop_small + len_k_outlierdrop_large} outliers removed out of {len_k_initial} rows:")
print(f" - Too small: {len_k_outlierdrop_small} rows removed")
print(f" - Too large: {len_k_outlierdrop_large} rows removed")

# Display the top rows sorted by 'modif_z_score'
df_k.sort_values('modif_z_score', ascending=False).head(2)


# In[330]:


df_k[df_k['country'].isin(developed_countries)].sort_values('modif_z_score_signed', ascending=True).head(10)


# ### DE-IT

# In[332]:


# grouped DF
df_it_g = df_it_uni.groupby(['seniority_level'])

# adding Z-scores
df_it_gz = df_it_g.apply(lambda x: add_z_scores(x, 'salary_normmed_2024_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_it_gz.sort_values('modif_z_score', ascending=False).head()


# In[333]:


# Sorting the dataframes for plotting
df_it_gzs = df_it_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_it_gmzs = df_it_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_it_gmzs_signed = df_it_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_it_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_it_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_it_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# In[334]:


# Start with the initial DataFrame
len_it_initial = len(df_it_gz)

# Define thresholds
lower_threshold = -3.0
upper_threshold = 3.0

# Boolean masks for outliers
mask_it_small = df_it_gz['modif_z_score_signed'] < lower_threshold
mask_it_large = df_it_gz['modif_z_score_signed'] > upper_threshold

# Count outliers
len_it_outlierdrop_small = mask_it_small.sum()
len_it_outlierdrop_large = mask_it_large.sum()

# Remove outliers
df_it = df_it_gz[~(mask_it_small | mask_it_large)].copy()

# Print results
print(f"{len_it_outlierdrop_small + len_it_outlierdrop_large} outliers removed out of {len_it_initial} rows:")
print(f" - Too small: {len_it_outlierdrop_small} rows removed")
print(f" - Too large: {len_it_outlierdrop_large} rows removed")

# Display the top 2 rows sorted by 'modif_z_score'
df_it.sort_values('modif_z_score', ascending=False).head(2)


# ### AI-Jobs.net

# In[336]:


# grouped DF
df_ai_g = df_ai_uni.groupby(['seniority_level'])

# adding Z-scores
df_ai_gz = df_ai_g.apply(lambda x: add_z_scores(x, 'salary_normmed_2024_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_ai_gz.sort_values('modif_z_score_signed', ascending=True).head()


# In[337]:


# Sorting the dataframes for plotting
df_ai_gzs = df_ai_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_ai_gmzs = df_ai_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_ai_gmzs_signed = df_ai_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_ai_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_ai_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_ai_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# In[338]:


# Start with the initial DataFrame
len_ai_initial = len(df_ai_gz)

# Define thresholds
lower_threshold = -3.0
upper_threshold = 3.0

# Boolean masks for outliers
mask_ai_small = df_ai_gz['modif_z_score_signed'] < lower_threshold
mask_ai_large = df_ai_gz['modif_z_score_signed'] > upper_threshold

# Count outliers
len_ai_outlierdrop_small = mask_ai_small.sum()
len_ai_outlierdrop_large = mask_ai_large.sum()

# Remove outliers
df_ai = df_ai_gz[~(mask_ai_small | mask_ai_large)].copy()

# Print results
print(f"{len_ai_outlierdrop_small + len_ai_outlierdrop_large} outliers removed out of {len_ai_initial} rows:")
print(f" - Too small: {len_ai_outlierdrop_small} rows removed")
print(f" - Too large: {len_ai_outlierdrop_large} rows removed")

# Display the top 2 rows sorted by 'modif_z_score'
df_ai.sort_values('modif_z_score', ascending=False).head(2)


# In[339]:


df_k[df_k['country'].isin(developed_countries)].sort_values('modif_z_score_signed', ascending=True).head(20)


# In[340]:


df_ai.sort_values('modif_z_score', ascending=False).head(5)


# In[341]:


df_it.sort_values('modif_z_score', ascending=False).head(5)


# # Data Quality Metrics

# ## Germany-It Survey

# In[344]:


cleaning_steps = [
    'Salary-nulls',
    'Employment-nulls',
    'Students',  
    'Never have coded',
    'Country-nulls',
    'Seniority-nulls',
    'Job-title-nulls',
    'Outliers: Too small salary',
    'Outliers: Too large salary'
]


# In[345]:


len_it_ini = sum([
    len_it18_ini,
    len_it19_ini,
    len_it20_ini,
    len_it21_ini,
    len_it22_ini,
    len_it23_ini
])

len_it18_salarydrop = len_it18_salarydrop1 - len_it18_salarydrop2
len_it19_salarydrop = len_it19_salarydrop1 - len_it19_salarydrop2
len_it20_salarydrop = len_it20_salarydrop1 - len_it20_salarydrop2
len_it21_salarydrop = len_it21_salarydrop1 - len_it21_salarydrop2
len_it22_salarydrop = len_it22_salarydrop1 - len_it22_salarydrop2
len_it23_salarydrop = len_it23_salarydrop1 - len_it23_salarydrop2

len_it_salarydrop = sum([
    len_it18_salarydrop,
    len_it19_salarydrop,
    len_it20_salarydrop,
    len_it21_salarydrop,
    len_it22_salarydrop,
    len_it23_salarydrop
])

len_it_employmentdrop  = len_it_employmentdrop1  - len_it_employmentdrop2
len_it_studentdrop     = len_it_studentdrop1     - len_it_studentdrop2
len_it_noncoderdrop    = 0
len_it_countrydrop     = len_it_countrydrop1     - len_it_countrydrop2
len_it_senioritydrop   = len_it_senioritydrop1   - len_it_senioritydrop2
len_it_jobtitledrop    = len_it_jobtitledrop1    - len_it_jobtitledrop2
#len_it_outlierdrop     = len_it_outlierdrop1     - len_it_outlierdrop2
#len_it_outlierdropnorm = len_it_outlierdropnorm1 - len_it_outlierdropnorm2

len_it_clean = (len(df_it))

it_difference  = len_it_ini - len_it_clean
it_cleanedaway = [
    len_it_salarydrop,
    len_it_employmentdrop,
    len_it_studentdrop,   
    len_it_noncoderdrop,
    len_it_countrydrop,
    len_it_senioritydrop,
    len_it_jobtitledrop,
    len_it_outlierdrop_small,
    len_it_outlierdrop_large
]


# In[346]:


print(f'Initial survey length: {len_it_ini}')
# Printing each variable in the list
for idx, value in enumerate(it_cleanedaway):
    print(f'Cleaning step {idx + 1}: {value}')
    
print(f'Final survey length: {len_it_clean}')
print(f'The difference between final and initial: {it_difference}')
print(f'Summing the individual cleaning steps: {sum(it_cleanedaway)}')


# ## Kaggle

# In[348]:


len_k_ini = sum([
    len_k19_ini,
    len_k20_ini,
    len_k21_ini,
    len_k22_ini
])

len_k19_salarydrop = len_k19_salarydrop1 - len_k19_salarydrop2
len_k20_salarydrop = len_k20_salarydrop1 - len_k20_salarydrop2
len_k21_salarydrop = len_k21_salarydrop1 - len_k21_salarydrop2
len_k22_salarydrop = len_k22_salarydrop1 - len_k22_salarydrop2

len_k_salarydrop = sum([
    len_k19_salarydrop,
    len_k20_salarydrop,
    len_k21_salarydrop,
    len_k22_salarydrop
])

len_k_employmentdrop  = 0
len_k_studentdrop     = 0
len_k_noncoderdrop    = len_k_noncoderdrop1    - len_k_noncoderdrop2
len_k_countrydrop     = len_k_countrydrop1     - len_k_countrydrop2
len_k_senioritydrop   = len_k_senioritydrop1   - len_k_senioritydrop2
len_k_jobtitledrop    = len_k_jobtitledrop1    - len_k_jobtitledrop2
#len_k_outlierdrop     = len_k_outlierdrop1     - len_k_outlierdrop2
#len_k_outlierdropnorm = len_k_outlierdropnorm1 - len_k_outlierdropnorm2

len_k_clean = (len(df_k))

k_difference  = len_k_ini - len_k_clean
k_cleanedaway = [
    len_k_salarydrop,
    len_k_employmentdrop,
    len_k_studentdrop,   
    len_k_noncoderdrop,
    len_k_countrydrop,
    len_k_senioritydrop,
    len_k_jobtitledrop,
    len_k_outlierdrop_small,
    len_k_outlierdrop_large
]

k_salarydrops = [
    len_k19_salarydrop,
    len_k20_salarydrop,
    len_k21_salarydrop,
    len_k22_salarydrop
]


# In[349]:


print(f'Initial survey length: {len_k_ini}')
# Printing each variable in the list
for idx, value in enumerate(k_cleanedaway):
    print(f'Cleaning step {idx + 1}: {value}')
    
print(f'Final survey length: {len_k_clean}')
print(f'The difference between final and initial: {k_difference}')
print(f'Summing the individual cleaning steps: {sum(k_cleanedaway)}')


# ## AI-Jobs.net

# In[351]:


#len_ai_ini
len_ai_salarydrop = 0

len_ai_employmentdrop  = len_ai_employmentdrop1  - len_ai_employmentdrop2
len_ai_studentdrop     = len_ai_studentdrop1     - len_ai_studentdrop2
len_ai_noncoderdrop    = 0
len_ai_countrydrop     = len_ai_countrydrop1     - len_ai_countrydrop2
len_ai_senioritydrop   = len_ai_senioritydrop1   - len_ai_senioritydrop2
len_ai_jobtitledrop    = len_ai_jobtitledrop1    - len_ai_jobtitledrop2
#len_ai_outlierdrop     = len_ai_outlierdrop1     - len_ai_outlierdrop2
#len_ai_outlierdropnorm = len_ai_outlierdropnorm1 - len_ai_outlierdropnorm2

len_ai_clean = (len(df_ai))

ai_difference  = len_ai_ini - len_ai_clean
ai_cleanedaway = [
    len_ai_salarydrop,
    len_ai_employmentdrop,
    len_ai_studentdrop,   
    len_ai_noncoderdrop,
    len_ai_countrydrop,
    len_ai_senioritydrop,
    len_ai_jobtitledrop,
    len_ai_outlierdrop_small,
    len_ai_outlierdrop_large
]


# In[352]:


print(f'Initial survey length: {len_ai_ini}')
# Printing each variable in the list
for idx, value in enumerate(ai_cleanedaway):
    print(f'Cleaning step {idx + 1}: {value}')
    
print(f'Final survey length: {len_ai_clean}')
print(f'The difference between final and initial: {ai_difference}')
print(f'Summing the individual cleaning steps: {sum(ai_cleanedaway)}')


# ## Plots

# In[354]:


# Data to plot
labels = cleaning_steps
sizes = it_cleanedaway

# Filter out zero segments
non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
filtered_labels = [labels[i] for i in non_zero_indices]
filtered_sizes = [sizes[i] for i in non_zero_indices]

# Create an explode list to separate slices for better readability
explode = [0.1] * len(filtered_sizes)  # Explode all slices for visibility

# Create the pie chart
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    filtered_sizes,
    #explode=explode,
    labels=filtered_labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=140,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1,  # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Germany-IT survey: ratio of cleaned-away data')
plt.show()


# In[355]:


# Data to plot
labels = cleaning_steps
sizes = ai_cleanedaway

# Filter out zero segments
non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
filtered_labels = [labels[i] for i in non_zero_indices]
filtered_sizes = [sizes[i] for i in non_zero_indices]

# Create an explode list to separate slices for better readability
explode = [0.1] * len(filtered_sizes)  # Explode all slices for visibility

# Create the pie chart
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    filtered_sizes,
    #explode=explode,
    labels=filtered_labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=140,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2) 
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('AI-Jobs.net: ratio of cleaned-away data')
plt.show()


# In[356]:


# Data to plot
labels = cleaning_steps
sizes = k_cleanedaway

# Filter out zero segments
non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
filtered_labels = [labels[i] for i in non_zero_indices]
filtered_sizes = [sizes[i] for i in non_zero_indices]

# Create an explode list to separate slices for better readability
explode = [0.1] * len(filtered_sizes)  # Explode all slices for visibility

# Create the pie chart
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    filtered_sizes,
    #explode=explode,
    labels=filtered_labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1,  # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2) 
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Kaggle: ratio of cleaned-away data')
plt.show()


# In[357]:


# Data to plot
labels = [2019,2020,2021,2022]
sizes = k_salarydrops

# Filter out zero segments
non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
filtered_labels = [labels[i] for i in non_zero_indices]
filtered_sizes = [sizes[i] for i in non_zero_indices]

# Create an explode list to separate slices for better readability
explode = [0.1] * len(filtered_sizes)  # Explode all slices for visibility

# Create the pie chart
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    filtered_sizes,
    #explode=explode,
    labels=filtered_labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=140,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1,  # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2) 
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('IT Industry Drops')
plt.show()


# In[358]:


# Data
categories = ['Germany IT-Survey', 'Kaggle', 'Ai-Jobs']
data_points_1 = [len_it_clean, sum(it_cleanedaway)]
data_points_2 = [len_k_clean, sum(k_cleanedaway)]
data_points_3 = [len_ai_clean, sum(ai_cleanedaway)]

# Create the figure and axis
fig, ax = plt.subplots()
bar_width = 0.35  # Width of the bars

# Colors for the stacks
colors = ['green', 'red']
colors = ['darkgreen', 'firebrick']
colors = ['olive', 'maroon']
colors = ['forestgreen', 'crimson']
colors = ['seagreen', 'darkred']
colors = ['teal', 'darkred']
colors = ['seagreen', 'maroon']

# Plot each bar category
bottom = [0] * len(categories)
for i, category in enumerate(categories):
    for j, value in enumerate([data_points_1, data_points_2, data_points_3][i]):
        ax.bar(category, value, bar_width, bottom=bottom[i], color=colors[j])
        bottom[i] += value

# Adding labels and title
ax.set_ylabel('Responses')
ax.set_title('Discarded Responses from the Surveys')

# Create custom legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
labels = ['Remaining Data', 'Cleaned-away Data']
ax.legend(handles, labels, loc='upper right')

# Show plot
plt.show()


# # Exporting the cleaned data

# In[398]:


# Assuming your current working directory is the 'notebooks' folder
import os

# Dictionary of DataFrames and their corresponding filenames
dataframes = {
    'df_k': df_k,
    'df_it': df_it,
    'df_ai': df_ai
    # Add more DataFrames and filenames as needed
}

df_combined = pd.concat(dataframes.values(), ignore_index=True)

# Path to the cleaned data folder
cleaned_data_folder = '../data/cleaned/'

# Ensure the folder exists
os.makedirs(cleaned_data_folder, exist_ok=True)

# Loop through the dictionary and export each DataFrame to a CSV file
for filename, dataframe in dataframes.items():
    full_path = os.path.join(cleaned_data_folder, f'{filename}.csv')
    dataframe.to_csv(full_path, index=False)

# Export the combined DataFrame to CSV
combined_path = os.path.join(cleaned_data_folder, 'df_combined.csv')
df_combined.to_csv(combined_path, index=False)

print(f'Individual DataFrames and combined DataFrame exported to {cleaned_data_folder}')


# In[400]:


# Dictionary of DataFrames and their corresponding filenames
dataframes = {
    'df_k': df_k,
    'df_it': df_it,
    'df_ai': df_ai
    # Add more DataFrames and filenames as needed
}

# Combine DataFrames into one
df_combined = pd.concat(dataframes.values(), ignore_index=True)

# Path to the cleaned data folder
cleaned_data_folder = '../data/cleaned/'

# Ensure the folder exists
os.makedirs(cleaned_data_folder, exist_ok=True)

# Export the combined DataFrame to CSV
combined_path = os.path.join(cleaned_data_folder, 'df_combined_tableau.csv')
df_combined.to_csv(combined_path, index=False, encoding='utf-8')

print(f'Individual DataFrames and combined DataFrame exported to {cleaned_data_folder}')


# # Draft

# ## Based on log-transformed salary

# ### AI-Jobs.net

# In[364]:


# grouped DF
df_ai_g = df_ai_uni.groupby(['seniority_level', 'country', 'year'])

# adding Z-scores
df_ai_gz = df_ai_g.apply(lambda x: add_z_scores(x, 'salary_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_ai_gz.sort_values('modif_z_score', ascending=False).head(10)


# In[365]:


# Sorting the dataframes for plotting
df_ai_gzs = df_ai_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_ai_gmzs = df_ai_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_ai_gmzs_signed = df_ai_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_ai_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_ai_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_ai_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# Why Use Different Thresholds? \
# Asymmetry in Data:
# - The distribution is not symmetric around the median.
# - High salaries are more spread out than low salaries.\
# Validity of High Salaries:
# - High salaries might be genuine and should not be automatically considered outliers.\
# No Negative Salaries:
# - Since salaries cannot be negative, the lower bound is naturally limited.

# In[367]:


len_ai_outlierdrop1 = len(df_ai_uni)

# Define asymmetric thresholds for outliers using Modified Z-score (Signed)
lower_threshold = -3  # Threshold for low outliers (negative side)
upper_threshold = 10  # Threshold for high outliers (positive side)

# Identify outliers and create new DataFrame without outliers
df_ai = df_ai_gz[
    (df_ai_gz['modif_z_score_signed'] >= lower_threshold) &
    (df_ai_gz['modif_z_score_signed'] <= upper_threshold)
].copy()

# Count outliers removed
num_outliers_removed = df_ai_gz.shape[0] - df_ai.shape[0]
len_ai_outlierdrop2 = len(df_ai)

# Print number of outliers removed
print(f"Number of outliers removed: {num_outliers_removed} out of {df_ai_gz.shape[0]} rows")
df_ai.sort_values('modif_z_score', ascending=False).head(4)


# ### TODO: --> TDD

# In[369]:


# Example DataFrame with more rows to reflect realistic salaries
# data = {
#     'seniority_level': ['Junior', 'Senior', 'Manager', 'Junior', 'Senior', 'Junior', 'Senior', 'Manager', 'Junior', 
#                         'Lead', 'Lead', 'Junior', 'Manager', 'Senior', 'Junior', 'Senior'],
#     'country': ['USA', 'USA', 'USA', 'USA', 'USA', 'Hungary', 'Hungary', 'Hungary', 'Hungary', 
#                 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany', 'Germany'],
#     'year': [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 
#              2020, 2020, 2020, 2020, 2020, 2020, 2020],
#     'salary': [60000, 80000, 120000, 55000, 75000, 40000, 60000, 100000, 2000000, 
#                342000, 345000, 30000, 250000, 200000, 32000, 195000]
# }
# 
# df = pd.DataFrame(data)
# 
# 
# def detect_outliers(data):
#     threshold = 15  # Adjust threshold as needed
#     median = data.median()
#     median_absolute_deviation = np.median(np.abs(data - median))
#     modified_z_scores = 0.6745 * np.abs(data - median) / median_absolute_deviation
#     return modified_z_scores > threshold
# 
# grouped = df.groupby(['seniority_level', 'country', 'year'])
# outliers = grouped['salary'].apply(detect_outliers)
# filtered_data = df[~outliers]
# 
# print(filtered_data)


# ### Germany IT-Survey

# In[371]:


# grouped DF
df_it_g = df_it_uni.groupby(['seniority_level', 'country', 'year'])

# adding Z-scores
df_it_gz = df_it_g.apply(lambda x: add_z_scores(x, 'salary_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_it_gz.sort_values('modif_z_score', ascending=False).head()


# There is one datapoint that is such a blatant outlier, it will distort any further evaluation.
# It needs to be taken out.

# In[373]:


# Cutting out salaries that are larger than 10.000.000 USD...
df_it_gz = df_it_gz[df_it_gz['salary'] <= 10000000]


# In[374]:


# Sorting the dataframes for plotting
df_it_gzs = df_it_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_it_gmzs = df_it_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_it_gmzs_signed = df_it_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_it_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_it_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_it_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# In[375]:


len_it_outlierdrop1 = len(df_it_uni)

# Define asymmetric thresholds for outliers using Modified Z-score (Signed)
lower_threshold = -3  # Threshold for low outliers (negative side)
upper_threshold = 10  # Threshold for high outliers (positive side)

# Identify outliers and create new DataFrame without outliers
df_it = df_it_gz[
    (df_it_gz['modif_z_score_signed'] >= lower_threshold) &
    (df_it_gz['modif_z_score_signed'] <= upper_threshold)
].copy()

# Count outliers removed
num_outliers_removed = df_it_gz.shape[0] - df_it.shape[0]
len_it_outlierdrop2 = len(df_it)

# Print number of outliers removed
print(f"Number of outliers removed: {num_outliers_removed} out of {df_it_gz.shape[0]} rows")
df_it.sort_values('modif_z_score', ascending=False).head(4)


# ### Kaggle 

# In[377]:


# grouped DF
df_k_g = df_k_uni.groupby(['experience', 'country', 'year'])

# adding Z-scores
df_k_gz = df_k_g.apply(lambda x: add_z_scores(x, 'salary_log')).reset_index(drop=True)
# outputting with descending Modified-Z-score order
df_k_gz.sort_values('z_score', ascending=False).head(5)


# In[378]:


# Sorting the dataframes for plotting
df_k_gzs = df_k_gz['z_score'].sort_values(ascending=False).reset_index(drop=True)
df_k_gmzs = df_k_gz['modif_z_score'].sort_values(ascending=False).reset_index(drop=True)
df_k_gmzs_signed = df_k_gz['modif_z_score_signed'].sort_values(ascending=False).reset_index(drop=True)

# Plotting
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(df_k_gzs, marker='.', linestyle='-', color='b')
plt.title('Sorted Z-scores')
plt.xlabel('Index')
plt.ylabel('Z-score')

plt.subplot(1, 3, 2)
plt.plot(df_k_gmzs, marker='.', linestyle='-', color='r')
plt.title('Sorted Modified Z-scores (Absolute)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score')

plt.subplot(1, 3, 3)
plt.plot(df_k_gmzs_signed, marker='.', linestyle='-', color='g')
plt.title('Sorted Modified Z-scores (Signed)')
plt.xlabel('Index')
plt.ylabel('Modified Z-score (Signed)')

plt.tight_layout()
plt.show()


# In[379]:


len_k_outlierdrop1 = len(df_k_uni)

# Define asymmetric thresholds for outliers using Modified Z-score (Signed)
lower_threshold = -3  # Threshold for low outliers (negative side)
upper_threshold = 10  # Threshold for high outliers (positive side)

# Identify outliers and create new DataFrame without outliers
df_k = df_k_gz[
    (df_k_gz['modif_z_score_signed'] >= lower_threshold) &
    (df_k_gz['modif_z_score_signed'] <= upper_threshold)
].copy()

# Count outliers removed
num_outliers_removed = df_k_gz.shape[0] - df_k.shape[0]
len_k_outlierdrop2 = len(df_k)

# Print number of outliers removed
print(f"Number of outliers removed: {num_outliers_removed} out of {df_k_gz.shape[0]} rows")
df_k.sort_values('modif_z_score', ascending=False).head(4)


# In[380]:


df_k_gz[~np.isfinite(df_k_gz['salary'])]


# In[381]:


df_k_gz[df_k_gz['salary'].isna()]


# ## Near-zero salaries in the developed world

# In[383]:


df_k[(df_k['seniority_level']=='senior') & (df_k['country'].isin(developed_countries))].sort_values(by='modif_z_score_signed').head()


# In[384]:


df_ai[(df_ai['seniority_level']=='senior') & (df_ai['country'].isin(developed_countries))].sort_values(by='modif_z_score_signed').head()


# In[385]:


df_it[(df_it['seniority_level']=='senior') & (df_it['country']=='de')].sort_values(by='modif_z_score_signed').head()


# These are nonsensical datapoints... 50 year old Data Engineer in the US with 5-10 years of experience... And with a 0-999 USD salary?! \
# I take an arbitrary step: I assume that no individual in the developing world in Junior/Medior/Senior/Executive positions supposed to have 0-999 USD salary range.\
# (interns are not included in this study) \
# Most probably these individuals gave their salary as a monthly value. And in each year's survey we have such individuals \
# But I throw these answers out.

# In[387]:


len(df_k[(df_k['seniority_level']=='senior') & 
          (df_k['country'].isin(developed_countries)) & 
          (df_k['salary'] < 1000)]['year'])


# In[388]:


len(df_k)


# In[389]:


df_k_test = df_k[((df_k['country'].isin(developed_countries)) & (df_k['salary'] > 10000) ) | ((~df_k['country'].isin(developed_countries)))]


# In[390]:


len(df_k_test)


# In[391]:


df_k_test[(df_k_test['seniority_level']=='senior') & (df_k_test['country']=='us')].sort_values(by='salary').head()


# In[ ]:




