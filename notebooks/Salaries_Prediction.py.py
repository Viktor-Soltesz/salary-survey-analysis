#!/usr/bin/env python
# coding: utf-8

# <font size="6.5"><b>Predicting Salaries data</b></font>

# <h1 style="background-color: #0e2e3b; color: white; font-size: 40px; font-weight: bold; padding: 10px;"> Import libraries & data, general settings </h1>

# In[2]:


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


# In[3]:


df_it = pd.read_csv('../data/cleaned/df_it.csv', low_memory=False)
df_k = pd.read_csv('../data/cleaned/df_k.csv', low_memory=False)
df_ai = pd.read_csv('../data/cleaned/df_ai.csv', low_memory=False)


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> Styles </h2>

# In[7]:


# General Display settings

# Column display is supressed by default
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
#df.head(200)

#changing the display format
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Plotting format
#print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')


# # Preparing Dataframes

# ## Western & Developed countries

# In[14]:


western_countries = [
    'al', 'ad', 'am', 'at', 'az', 'by', 'be', 'ba', 'bg', 'hr',
    'cy', 'cz', 'dk', 'ee', 'fi', 'fr', 'ge', 'de', 'gr', 'hu',
    'is', 'ie', 'it', 'kz', 'xk', 'lv', 'li', 'lt', 'lu', 'mt',
    'md', 'mc', 'me', 'nl', 'mk', 'no', 'pl', 'pt', 'ro', 'ru',
    'sm', 'rs', 'sk', 'si', 'es', 'se', 'ch', 'tr', 'ua', 'gb',
    'va', 'ca', 'au', 'us'
]


# In[16]:


western_countries = ['de', 'gb', 'nl', 'se', 'dk', 'be', 'fi', 'at', 'ch', 'ie', 'ca', 'au', 'us']


# In[18]:


df_ai_w = df_ai.copy()
df_ai_w['country'] = df_ai_w['country'][df_ai_w['country'].isin(western_countries)]

df_k_w = df_k.copy()
df_k_w['country'] = df_k_w['country'][df_k_w['country'].isin(western_countries)]


# In[20]:


# df_ai_d = df_ai.copy()
# df_ai_d['country'] = df_ai_d['country'][df_ai_d['country'].isin(developed_countries)]
# 
# df_k_d = df_k.copy()
# df_k_d['country'] = df_k_d['country'][df_k_d['country'].isin(developed_countries)]


# In[22]:


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


# ## Glassdoor salary values

# In[25]:


glassdoor_data = pd.DataFrame({
    'job_category': ['Data Analyst', 'Data Analyst', 'Data Engineer', 'Data Engineer', 'Software Engineer'],
    'seniority_level': ['junior', 'senior', 'senior', 'senior', 'senior'],
    'country': ['us', 'de', 'us', 'us', 'de'],
    'glassdoor_lower': [50000, 73000, 60000, 80000, 65000],
    'glassdoor_upper': [70000, 108000, 80000, 100000, 90000]
})


# In[27]:


# Define the column names
columns = ['job_category', 'seniority_level', 'country', 'glassdoor_lower', 'glassdoor_upper']

# Create a list of tuples
glassdoor_data_rows = [
    ('Data Analyst', 'junior', 'us', 81000, 134000),
    ('Data Analyst', 'medior', 'us', 86000, 144000),
    ('Data Analyst', 'senior', 'us', 88000, 150000),
    ('Data Analyst', 'junior', 'de', 55000, 71000),
    ('Data Analyst', 'medior', 'de', 58000, 82000),
    ('Data Analyst', 'senior', 'de', 68000, 93000),
    ('Data Engineer', 'junior', 'us', 96000, 157000),
    ('Data Engineer', 'medior', 'us', 109000, 179000),
    ('Data Engineer', 'senior', 'us', 121000, 198000),
    ('Data Engineer', 'junior', 'de', 62500, 81000),
    ('Data Engineer', 'medior', 'de', 69000, 90100),
    ('Data Engineer', 'senior', 'de', 73500, 94000),
    ('Data Scientist/ ML Engineer', 'junior', 'us', 129000, 210000),
    ('Data Scientist/ ML Engineer', 'medior', 'us', 140000, 232000),
    ('Data Scientist/ ML Engineer', 'senior', 'us', 149000, 251000),
    ('Data Scientist/ ML Engineer', 'junior', 'de', 63500, 83000),
    ('Data Scientist/ ML Engineer', 'medior', 'de', 74500, 98500),
    ('Data Scientist/ ML Engineer', 'senior', 'de', 72000, 104000),
]

# Create the DataFrame
glassdoor_data = pd.DataFrame.from_records(glassdoor_data_rows, columns=columns)


# In[29]:


# Define the column names
columns = ['job_category', 'seniority_level', 'country', 'glassdoor_lower', 'glassdoor_upper']

# Create a list of tuples
glassdoor_data_rows = [
    ('Data Analyst', 'junior', 'us', 70000, 117000),
    ('Data Analyst', 'medior', 'us', 81000, 134000),
    ('Data Analyst', 'senior', 'us', 88000, 150000),
    
    ('Data Analyst', 'junior', 'de', 51500, 70000),
    ('Data Analyst', 'medior', 'de', 55000, 71000),
    ('Data Analyst', 'senior', 'de', 68000, 93000),
    
    ('Data Engineer', 'junior', 'us', 84000, 142000),
    ('Data Engineer', 'medior', 'us', 96000, 157000),
    ('Data Engineer', 'senior', 'us', 121000, 198000),
    
    ('Data Engineer', 'junior', 'de', 57000, 74500),
    ('Data Engineer', 'medior', 'de', 62500, 81000),
    ('Data Engineer', 'senior', 'de', 73500, 94000),
    
    ('Data Scientist/ ML Engineer', 'junior', 'us', 117000, 196000),
    ('Data Scientist/ ML Engineer', 'medior', 'us', 129000, 210000),
    ('Data Scientist/ ML Engineer', 'senior', 'us', 149000, 251000),
    
    ('Data Scientist/ ML Engineer', 'junior', 'de', 58000, 79000),
    ('Data Scientist/ ML Engineer', 'medior', 'de', 63500, 83000),
    ('Data Scientist/ ML Engineer', 'senior', 'de', 72000, 104000),
]

# Create the DataFrame
glassdoor_data = pd.DataFrame.from_records(glassdoor_data_rows, columns=columns)


# ## Exclusion categories

# In[32]:


seniority_exclusion = ['other'] #executive

exclude_categories = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers', 'Other managers']
jobcategory_exclusion = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers']


# In[34]:


# First we filter for western countries
df_ai_w= df_ai.copy()
df_it_w= df_it.copy()
df_k_w = df_k.copy()

# Western countries
df_ai_w = df_ai_w[df_ai_w['country'].isin(western_countries)].copy()
df_it_w = df_it_w[df_it_w['country'].isin(western_countries)].copy()
df_k_w = df_k_w[df_k_w['country'].isin(western_countries)].copy()

# Filter out rows where 'job_category' is in the exclude list
df_ai_w = df_ai_w[~df_ai_w['seniority_level'].isin(seniority_exclusion)]
df_it_w = df_it_w[~df_it_w['seniority_level'].isin(seniority_exclusion)]
df_k_w = df_k_w[~df_k_w['seniority_level'].isin(seniority_exclusion)]

df_ai_w = df_ai_w[~df_ai_w['job_category'].isin(jobcategory_exclusion)]
df_it_w = df_it_w[~df_it_w['job_category'].isin(jobcategory_exclusion)]
df_k_w = df_k_w[~df_k_w['job_category'].isin(jobcategory_exclusion)]


# In[36]:


# Filtering out rows, which would result in Factorial Groups with less datapoint than the statistically required threshold

# Specify the minimum number of counts required to keep the group
min_count = 20

# Calculate the count of data points for each job category
group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()

# Rename the count column for clarity
group_counts1.rename(columns={'salary_norm': 'count'}, inplace=True)
group_counts2.rename(columns={'salary_norm': 'count'}, inplace=True)
group_counts3.rename(columns={'salary_norm': 'count'}, inplace=True)

# Filter groups that meet the criteria
valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
valid_groups3 = group_counts3[group_counts3['count'] >= min_count]

# Merge the valid groups back with the original dataframe to keep only the desired rows
df_ai_w_l = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
                       on=['seniority_level', 'job_category'], 
                       how='inner')

df_it_w_l = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
                       on=['seniority_level', 'job_category'], 
                       how='inner')

df_k_w_l = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
                       on=['seniority_level', 'job_category'], 
                       how='inner')


# In[38]:


df_combined = pd.concat([df.dropna(axis=1, how='all') for df in [df_ai_w_l, df_it_w_l, df_k_w_l]])


# ## Data-professionals

# In[41]:


data_fields = ['Data Analyst', 'Data Engineer', 'Data Scientist/ ML Engineer']

df_k_w_data =  df_k_w[df_k_w['job_category'].isin(data_fields)]
df_it_w_data = df_it[df_it['job_category'].isin(data_fields)]
df_ai_w_data = df_ai_w[df_ai_w['job_category'].isin(data_fields)]


# In[43]:


df_k_w_data = df_k_w_data[(df_k_w_data['seniority_level'] != 'executive')]
df_it_w_data = df_it_w_data[(df_it_w_data['seniority_level'] != 'executive')]
df_it_w_data = df_it_w_data[(df_it_w_data['seniority_level'] != 'other')]
df_ai_w_data = df_ai_w_data[(df_ai_w_data['seniority_level'] != 'executive')]


# # CV in each factorial cell to get a feeling for RMSE and R-squared

# CV (Coefficient of Variation) = Standard Deviation / Mean salary

# In[47]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_ai.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_ai.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(12, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# In[48]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_ai_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_ai_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(12, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# In[50]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_k.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_k.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(12, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# In[51]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_k_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_k_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(12, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# In[53]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_it.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_it.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(16, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 6})
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# In[55]:


# Create a pivot table to calculate the mean salary in each factorial cell
pivot_table_mean = df_it_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='mean', fill_value=0)

# Create a pivot table to calculate the standard deviation of salary in each factorial cell
pivot_table_std = df_it_w_l.pivot_table(index=['seniority_level'], columns='job_category', values='salary', aggfunc='std', fill_value=0)

# Calculate the Coefficient of Variation (CV) by dividing std by the mean
pivot_table_cv = pivot_table_std / pivot_table_mean

# Plot the heatmap of CV
plt.figure(figsize=(16, 3))
sns.heatmap(pivot_table_cv, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 6})
plt.title('Coefficient of Variation (CV) of Salary in Each Factorial Cell for Survey K')
plt.xlabel('Job Category')
plt.ylabel('Seniority')
plt.show()


# # Multiregression

# In[71]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from patsy import dmatrices
from scipy.stats import shapiro
from scipy.stats import lognorm

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ### Functions

# In[74]:


def train_model_2301(X, y):
    """
    Trains a multilinear regression model using statsmodels, with one-hot encoding for categorical variables.
    Handles both numerical and categorical variables properly.
    
    Parameters:
    X (pd.DataFrame): Independent variables (can contain both categorical and numerical columns)
    y (pd.Series or pd.DataFrame): Dependent variable
    
    Returns:
    model: Fitted statsmodels regression model
    encoded_features: Dictionary containing feature names and their encodings for future use
    X_encoded: The encoded independent variables used in the model
    y: The dependent variable aligned with X_encoded
    """
    # Create copies to avoid modifying the original data
    X_copy = X.copy()
    y_copy = y.copy()
    
    # Dictionary to store category mappings
    encoded_features = {}
    
    # Process each column
    for column in X_copy.columns:
        # Check if column is categorical (object dtype) or contains strings
        if X_copy[column].dtype == 'object' or pd.api.types.is_string_dtype(X_copy[column]):
            # Store unique categories for this column
            encoded_features[column] = sorted(X_copy[column].unique().tolist())
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X_copy, drop_first=True, dtype=float)
    
    # Add constant term
    X_encoded = sm.add_constant(X_encoded)
    
    # Convert y to float type if it isn't already
    y_copy = y_copy.astype(float)
    
    # Reset indices to ensure alignment
    X_encoded.reset_index(drop=True, inplace=True)
    y_copy.reset_index(drop=True, inplace=True)
    
    # Fit the model
    model = sm.OLS(y_copy, X_encoded).fit()
    
    return model, encoded_features, X_encoded, y_copy


# In[76]:


import pandas as pd
import statsmodels.api as sm

def train_model_1617(X, y):
    """
    Trains a multilinear regression model using statsmodels, with one-hot encoding for categorical variables.
    Handles both numerical and categorical variables properly.
    
    Parameters:
    X (pd.DataFrame): Independent variables (can contain both categorical and numerical columns)
    y (pd.Series or pd.DataFrame): Dependent variable
    
    Returns:
    model: Fitted statsmodels regression model
    encoded_features: Dictionary containing feature names and their encodings for future use
    X_encoded: The encoded independent variables used in the model
    y: The dependent variable aligned with X_encoded
    """
    # Create copies to avoid modifying the original data
    X_copy = X.copy()
    y_copy = y.copy()
    
    # Dictionary to store category mappings
    encoded_features = {}
    
    # Process each column
    for column in X_copy.columns:
        # Check if column is categorical (object dtype) or contains strings
        if X_copy[column].dtype == 'object' or pd.api.types.is_string_dtype(X_copy[column]):
            # Fill NaN values with a placeholder
            X_copy[column] = X_copy[column].fillna('Missing')
            # Store unique categories for this column
            encoded_features[column] = sorted(X_copy[column].unique().tolist())
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X_copy, drop_first=True, dtype=float)
    
    # Add constant term
    X_encoded = sm.add_constant(X_encoded)
    
    # Convert y to float type if it isn't already
    y_copy = y_copy.astype(float)
    
    # Reset indices to ensure alignment
    X_encoded.reset_index(drop=True, inplace=True)
    y_copy.reset_index(drop=True, inplace=True)
    
    # Fit the model
    model = sm.OLS(y_copy, X_encoded).fit()
    
    return model, encoded_features, X_encoded, y_copy


# In[78]:


# Function to calculate leverage and Cook's Distance
def calculate_leverage_and_cooks_distance(model):
    # Get influence measures
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    return leverage, cooks_d

# Function to plot leverage and Cook's Distance
def plot_leverage_and_cooks_distance(leverage, cooks_d, leverage_threshold=None, cooks_threshold=None):
    n = len(leverage)
    p = model.df_model  # Number of predictors (including dummy variables)
    
    if leverage_threshold is None:
        leverage_threshold = 2 * (p + 1) / n
    if cooks_threshold is None:
        cooks_threshold = 4 / n
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 3))

    # Leverage Plot
    ax[0].scatter(range(n), leverage, alpha=0.5, s=5)
    ax[0].axhline(y=leverage_threshold, color='r', linestyle='--', label='Leverage Threshold')
    ax[0].set_ylim(0,0.06)
    ax[0].set_xlabel('Observation')
    ax[0].set_ylabel('Leverage')
    ax[0].set_title('Leverage Values')
    ax[0].legend()

    # Cook's Distance Plot
    ax[1].scatter(range(n), cooks_d, alpha=0.5, s=5)
    ax[1].axhline(y=cooks_threshold, color='r', linestyle='--', label="Cook's Distance Threshold")
    ax[1].set_ylim(0,0.005)
    ax[1].set_xlabel('Observation')
    ax[1].set_ylabel("Cook's Distance")
    ax[1].set_title("Cook's Distance Values")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# In[80]:


def plot_model_coefficients(model, sizex, sizey):
    """
    Extracts and plots the coefficients from a statsmodels OLS regression model, excluding the intercept.
    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): The fitted OLS model.
    """
    # Extract the coefficients and their names
    coefficients = model.params
    feature_names = coefficients.index
    
    # Create a DataFrame for coefficients
    coefficients_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients.values
    })

    # Remove the constant (intercept) term if it exists
    coefficients_df = coefficients_df[coefficients_df['Feature'] != 'const']
    
    # Sort the dataframe by coefficient values for better visualization
    coefficients_df_sorted = coefficients_df.sort_values(by='Coefficient')

    # Create a horizontal bar plot
    plt.figure(figsize=(sizex, sizey))
    sns.barplot(
        x='Coefficient',
        y='Feature',
        data=coefficients_df_sorted,
        hue='Feature',
        palette='coolwarm',
        orient='h',
        legend=False
    )

    # Add a vertical line at x=0 for reference
    plt.axvline(x=0, color='grey', linewidth=0.8)

    # Set plot title and labels
    plt.title('Regression Coefficients for Salary Prediction Model')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')

    # Adjust layout to ensure all labels are visible
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[82]:


def calculate_rmse_original_scale(model, y_true_log):
    """
    Calculates RMSE on the original scale for a model trained with log-transformed values.

    Parameters:
    - model: The fitted statsmodels OLS model.
    - y_true_log: The actual log-transformed values of the dependent variable.

    Returns:
    - rmse_original: RMSE calculated on the original (non-logarithmic) scale.
    """
    # Calculate predictions in the log scale
    y_pred_log = model.fittedvalues

    # Back-transform predictions and actuals to the original scale
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_true_log)

    # Calculate RMSE in the original scale
    rmse_original = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse_original


# In[84]:


def predict_salary_for_case_log(model, encoded_features, specific_case):
    """
    Predicts the salary for a specific case using the statsmodels OLS model.

    Parameters:
    - model: The fitted statsmodels OLS model.
    - encoded_features: Dictionary containing feature names and their encodings.
    - specific_case: Dictionary of feature values for the specific case.

    Returns:
    - predicted_salary: The predicted salary (transformed back from log scale if applicable).
    """
    # Create a DataFrame for the specific case
    specific_case_df = pd.DataFrame([specific_case])

    # Initialize all features with zero values
    all_features = model.params.index.tolist()
    specific_case_encoded = pd.DataFrame(columns=all_features)
    specific_case_encoded.loc[0] = 0  # Initialize all features to zero

    # Handle categorical variables
    for column in encoded_features.keys():
        categories = encoded_features[column]
        # Skip the reference category due to drop_first=True
        for cat in categories[1:]:
            col_name = f"{column}_{cat}"
            if specific_case[column] == cat:
                specific_case_encoded.at[0, col_name] = 1
            else:
                specific_case_encoded.at[0, col_name] = 0

    # Handle numerical variables (if any)
    numerical_columns = [col for col in specific_case.keys() if col not in encoded_features.keys()]
    for col in numerical_columns:
        specific_case_encoded.at[0, col] = specific_case[col]

    # Add constant term
    specific_case_encoded['const'] = 1.0  # Ensure the constant term is included

    # Reorder columns to match the model's parameters
    specific_case_encoded = specific_case_encoded[model.params.index]

    # Ensure all columns are float64
    specific_case_encoded = specific_case_encoded.astype('float64')

    # Make the prediction
    predicted_log_salary = model.predict(specific_case_encoded)[0]

    # If the model was trained on log-transformed salaries, exponentiate to get the actual salary
    # Adjust this line if your target variable is not log-transformed
    predicted_salary = np.exp(predicted_log_salary)

    return predicted_salary


# In[86]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

def predict_salary_for_case_log_perc(model, encoded_features, specific_case):
    """
    Predicts the salary and calculates the 25th and 75th percentiles for a specific case
    using the statsmodels OLS model trained on log-transformed salaries.

    Parameters:
    - model: The fitted statsmodels OLS model.
    - encoded_features: Dictionary containing feature names and their encodings.
    - specific_case: Dictionary of feature values for the specific case.

    Returns:
    - predicted_salary: The predicted median salary (original scale).
    - p25_salary: The 25th percentile of the predicted salary distribution (original scale).
    - p75_salary: The 75th percentile of the predicted salary distribution (original scale).
    """
    # Create a DataFrame for the specific case
    specific_case_df = pd.DataFrame([specific_case])

    # Initialize all features with zero values
    all_features = model.params.index.tolist()
    specific_case_encoded = pd.DataFrame(columns=all_features)
    specific_case_encoded.loc[0] = 0  # Initialize all features to zero

    # Handle categorical variables
    for column in encoded_features.keys():
        categories = encoded_features[column]
        # Skip the reference category due to drop_first=True
        for cat in categories[1:]:
            col_name = f"{column}_{cat}"
            if specific_case[column] == cat:
                specific_case_encoded.at[0, col_name] = 1
            else:
                specific_case_encoded.at[0, col_name] = 0

    # Handle numerical variables (if any)
    numerical_columns = [col for col in specific_case.keys() if col not in encoded_features.keys()]
    for col in numerical_columns:
        specific_case_encoded.at[0, col] = specific_case[col]

    # Add constant term
    specific_case_encoded['const'] = 1.0  # Ensure the constant term is included

    # Reorder columns to match the model's parameters
    specific_case_encoded = specific_case_encoded[model.params.index]

    # Ensure all columns are float64
    specific_case_encoded = specific_case_encoded.astype('float64')

    # Make the prediction in log scale
    predicted_log_salary = model.predict(specific_case_encoded)[0]

    # Obtain the standard deviation of the residuals (sigma_log)
    sigma_log = np.sqrt(model.scale)

    # Compute the 25th and 75th percentiles in log space
    p25_log_salary = predicted_log_salary + norm.ppf(0.25) * sigma_log
    p75_log_salary = predicted_log_salary + norm.ppf(0.75) * sigma_log

    # Exponentiate to get the salaries in original scale
    predicted_salary = np.exp(predicted_log_salary)
    p25_salary = np.exp(p25_log_salary)
    p75_salary = np.exp(p75_log_salary)

    return predicted_salary, p25_salary, p75_salary


# In[88]:


def predict_salary_for_case(model, encoded_features, specific_case):
    """
    Predicts the salary for a specific case using the statsmodels OLS model.

    Parameters:
    - model: The fitted statsmodels OLS model.
    - encoded_features: Dictionary containing feature names and their encodings.
    - specific_case: Dictionary of feature values for the specific case.

    Returns:
    - predicted_salary: The predicted salary (transformed back from log scale if applicable).
    """
    # Create a DataFrame for the specific case
    specific_case_df = pd.DataFrame([specific_case])

    # Initialize all features with zero values
    all_features = model.params.index.tolist()
    specific_case_encoded = pd.DataFrame(columns=all_features)
    specific_case_encoded.loc[0] = 0  # Initialize all features to zero

    # Handle categorical variables
    for column in encoded_features.keys():
        categories = encoded_features[column]
        # Skip the reference category due to drop_first=True
        for cat in categories[1:]:
            col_name = f"{column}_{cat}"
            if specific_case[column] == cat:
                specific_case_encoded.at[0, col_name] = 1
            else:
                specific_case_encoded.at[0, col_name] = 0

    # Handle numerical variables (if any)
    numerical_columns = [col for col in specific_case.keys() if col not in encoded_features.keys()]
    for col in numerical_columns:
        specific_case_encoded.at[0, col] = specific_case[col]

    # Add constant term
    specific_case_encoded['const'] = 1.0  # Ensure the constant term is included

    # Reorder columns to match the model's parameters
    specific_case_encoded = specific_case_encoded[model.params.index]

    # Ensure all columns are float64
    specific_case_encoded = specific_case_encoded.astype('float64')

    # Make the prediction
    predicted_salary = model.predict(specific_case_encoded)[0]
    
    return predicted_salary


# ### Plotting functions

# In[91]:


def plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6):
    """
    Plots a histogram with KDE and a Q-Q plot of residuals side by side.
    
    Parameters:
        residuals (pd.Series): Series of residuals to analyze.
        hist_bins (int): Number of bins for the histogram.
        qq_marker_size (float): Size of the dots in the Q-Q plot.
        qq_alpha (float): Alpha (transparency) of the dots in the Q-Q plot.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))

    # Plot histogram with KDE
    sns.histplot(residuals, kde=True, bins=hist_bins, ax=axes[0])
    axes[0].set_title('Histogram of Residuals')
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Frequency')

    # Create Q-Q plot
    qq_plot = sm.qqplot(residuals, line='s', ax=axes[1])
    plt.setp(qq_plot.gca().get_lines(), markersize=qq_marker_size, alpha=qq_alpha)
    axes[1].set_title('Q-Q Plot of Residuals')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# In[93]:


def simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=1000, dot_size=10):
    """
    Simulates Shapiro-Wilk tests on random subsamples of residuals over a range of sample sizes,
    calculating the average p-value across simulations for each sample size.
    
    Parameters:
        residuals (pd.Series): Series of residuals to sample from.
        min_sample_size (int): Minimum sample size for the subsampling.
        max_sample_size (int): Maximum sample size for the subsampling.
        step_size (int): Step size for increasing sample size.
        num_simulations (int): Number of simulations per sample size.
        dot_size (int): Size of the dots in the final plot.
    
    Returns:
        pd.DataFrame: DataFrame containing sample sizes and corresponding average Shapiro-Wilk p-values.
    """
    # Define the range of sample sizes
    sample_sizes = np.arange(min_sample_size, max_sample_size + 1, step_size)

    # Initialize a list to store the results
    results = []

    # Loop through each sample size
    for size in sample_sizes:
        p_values = []  # List to store p-values for each simulation at this sample size

        # Perform multiple simulations for each sample size
        for _ in range(num_simulations):
            # Take a random subsample of the given size from residuals
            subsample = residuals.sample(size, random_state=np.random.randint(0, 10000))
            
            # Perform Shapiro-Wilk test and store the p-value
            _, p_value = shapiro(subsample)
            p_values.append(p_value)
        
        # Calculate the average p-value across simulations
        avg_p_value = np.mean(p_values)
        
        # Append results to the list
        results.append({'Sample Size': size, 'Average Shapiro-Wilk p-value': avg_p_value})

    # Convert results to a DataFrame for easy plotting
    results_df = pd.DataFrame(results)

    # Plot the Sample Size vs. Average P-Value curve
    plt.figure(figsize=(12, 4))
    plt.scatter(results_df['Sample Size'], results_df['Average Shapiro-Wilk p-value'], alpha=0.5, s=dot_size, c="blue")
    plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
    plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
    plt.xlabel('Sample Size')
    plt.ylabel('Average Shapiro-Wilk p-value')
    plt.title('Sample Size vs. Average Shapiro-Wilk p-value (across simulations)')
    plt.legend()
    plt.show()

    return results_df


# In[95]:


# Plot residuals to check for homoscedasticity
def plot_homoscedasticity(residuals):
    
    # Scatter plot of residuals vs fitted values
    plt.scatter(fitted_values, residuals, alpha=0.05, s=5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()


# In[97]:


def simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=17000, step_size=100, num_simulations=100, dot_size=10):
    """
    Simulates Breusch-Pagan tests on random subsamples of residuals and fitted values over a range of sample sizes,
    calculating the average p-value across simulations for each sample size.
    
    Parameters:
        residuals (pd.Series): Series of residuals to sample from.
        fitted_values (pd.Series): Series of fitted values corresponding to the residuals.
        min_sample_size (int): Minimum sample size for the subsampling.
        max_sample_size (int): Maximum sample size for the subsampling.
        step_size (int): Step size for increasing sample size.
        num_simulations (int): Number of simulations per sample size.
        dot_size (int): Size of the dots in the final plot.
    
    Returns:
        pd.DataFrame: DataFrame containing sample sizes and corresponding average Breusch-Pagan p-values.
    """
    # Define the range of sample sizes
    sample_sizes = np.arange(min_sample_size, max_sample_size + 1, step_size)

    # Initialize a list to store the results
    results = []

    # Loop through each sample size
    for size in sample_sizes:
        p_values = []  # List to store p-values for each simulation at this sample size

        # Perform multiple simulations for each sample size
        for _ in range(num_simulations):
            # Randomly sample indices and select subset of residuals and fitted values
            indices = np.random.choice(range(len(residuals)), size=size, replace=False)
            subsample_residuals = residuals.iloc[indices]
            subsample_fitted_values = fitted_values.iloc[indices]

            # Prepare data for Breusch-Pagan test
            exog = add_constant(subsample_fitted_values)  # Add constant to independent variables
            bp_test = het_breuschpagan(subsample_residuals, exog)

            # Store the p-value from the Breusch-Pagan test
            p_value = bp_test[1]  # p-value is the second element in the returned tuple
            p_values.append(p_value)
        
        # Calculate the average p-value across simulations
        avg_p_value = np.mean(p_values)
        
        # Append results to the list
        results.append({'Sample Size': size, 'Average Breusch-Pagan p-value': avg_p_value})

    # Convert results to a DataFrame for easy plotting
    results_df = pd.DataFrame(results)

    # Plot the Sample Size vs. Average P-Value curve
    plt.figure(figsize=(10, 4))
    plt.scatter(results_df['Sample Size'], results_df['Average Breusch-Pagan p-value'], alpha=0.5, s=dot_size, c="purple")
    plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
    plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
    plt.xlabel('Sample Size')
    plt.ylabel('Average Breusch-Pagan p-value')
    plt.title('Sample Size vs. Average Breusch-Pagan p-value (across simulations)')
    plt.legend()
    plt.show()

    return results_df


# In[99]:


def plot_salary_distributions(df_name, model, encoded_features, specific_case, glassdoor_data):
    """
    Plots predicted and fitted lognormal salary distributions, observed salary data,
    and Glassdoor salary range (if available) for a specific case.

    Parameters:
        df_name (pd.DataFrame): DataFrame containing observed salary data.
        model: Model for predicting salary.
        encoded_features: Encoded features for prediction.
        specific_case (dict): Dictionary with keys 'job_category', 'seniority_level', and 'country'.
        glassdoor_data (pd.DataFrame): DataFrame with Glassdoor salary range data.

    Returns:
        None
    """
    # Extract case parameters
    job_category = specific_case['job_category']
    seniority_level = specific_case['seniority_level']
    country = specific_case['country']
    
    # Retrieve salary data for specific case
    salary_data = df_name['salary'][
        (df_name['seniority_level'] == seniority_level) &
        (df_name['job_category'] == job_category) &
        (df_name['country'] == country)
    ]
    
    # Fit a lognormal distribution to observed salary data
    shape_fit, loc_fit, scale_fit = lognorm.fit(salary_data, floc=0)
    mu_fit = np.log(scale_fit)
    sigma_fit = shape_fit
    
    # Predict salary and percentiles
    predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(
        model, encoded_features, specific_case
    )
    
    # Calculate predicted lognormal parameters
    z_25 = -0.67448975  # z-score for the 25th percentile
    z_75 = 0.67448975   # z-score for the 75th percentile
    sigma_pred = (np.log(predicted_salary_p75) - np.log(predicted_salary_p25)) / (z_75 - z_25)
    mu_pred = np.log(predicted_salary)
    
    # Set upper limit for x-axis
    P99_pred = lognorm.ppf(0.99, s=sigma_pred, scale=np.exp(mu_pred))
    P99_fitted = lognorm.ppf(0.99, s=sigma_fit, scale=np.exp(mu_fit))
    upper_limit = max(P99_pred, P99_fitted, salary_data.max()) * 1.1
    x = np.linspace(0.01, upper_limit, 200)
    
    # Generate lognormal PDFs
    predicted_pdf = lognorm.pdf(x, s=sigma_pred, scale=np.exp(mu_pred))
    fitted_pdf = lognorm.pdf(x, s=sigma_fit, scale=np.exp(mu_fit))
    
    # Calculate percentiles for predicted and fitted distributions
    P25_pred, P50_pred, P75_pred = predicted_salary_p25, predicted_salary, predicted_salary_p75
    P25_fitted = lognorm.ppf(0.25, s=sigma_fit, scale=np.exp(mu_fit))
    P50_fitted = lognorm.ppf(0.50, s=sigma_fit, scale=np.exp(mu_fit))
    P75_fitted = lognorm.ppf(0.75, s=sigma_fit, scale=np.exp(mu_fit))
    
    # Retrieve Glassdoor Salary Range
    glassdoor_row = glassdoor_data[
        (glassdoor_data['job_category'] == job_category) &
        (glassdoor_data['seniority_level'] == seniority_level) &
        (glassdoor_data['country'] == country)
    ]
    glassdoor_lower = glassdoor_row['glassdoor_lower'].values[0] if not glassdoor_row.empty else None
    glassdoor_upper = glassdoor_row['glassdoor_upper'].values[0] if not glassdoor_row.empty else None
    
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(x, predicted_pdf, label=f'Predicted Lognormal Distribution\n$\\mu={mu_pred:.2f}, \\sigma={sigma_pred:.2f}$', color='green')
    plt.plot(x, fitted_pdf, label=f'Fitted Lognormal Distribution\n$\\mu={mu_fit:.2f}, \\sigma={sigma_fit:.2f}$', color='blue')
    plt.hist(salary_data, bins=30, density=True, alpha=0.1, label='Observed Salary Data', color='blue')
    
    # Annotate percentiles for predicted distribution
    plt.axvline(P25_pred, color='green', alpha=0.7, linestyle=':', label=f'Predicted 25th percentile: {P25_pred:.2f}')
    plt.axvline(P50_pred, color='green', alpha=0.7, linestyle=':', label=f'Predicted 50th percentile (median): {P50_pred:.2f}')
    plt.axvline(P75_pred, color='green', alpha=0.7, linestyle=':', label=f'Predicted 75th percentile: {P75_pred:.2f}')
    
    # Annotate percentiles for fitted distribution
    plt.axvline(P25_fitted, color='blue', alpha=0.5, linestyle='--', label=f'Fitted 25th percentile: {P25_fitted:.2f}')
    plt.axvline(P50_fitted, color='blue', alpha=0.5, linestyle='--', label=f'Fitted 50th percentile (median): {P50_fitted:.2f}')
    plt.axvline(P75_fitted, color='blue', alpha=0.5, linestyle='--', label=f'Fitted 75th percentile: {P75_fitted:.2f}')
    
    # Add Glassdoor salary range if available
    if glassdoor_lower is not None and glassdoor_upper is not None:
        plt.axvspan(glassdoor_lower, glassdoor_upper, color='orange', alpha=0.25,
                    label=f'Glassdoor Salary Range: {glassdoor_lower} - {glassdoor_upper}')
    
    # Labels and title
    plt.title('Lognormal Distributions (Predicted and Fitted) and Observed Salary Data')
    plt.xlabel('Salary')
    plt.ylabel('Probability Density Function (PDF)')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[101]:


def plot_salary_percentiles_by_case(df_name, model, encoded_features, specific_cases, glassdoor_data):
    """
    Plots salary percentiles (P25, P50, P75) for different cases and data sources (Observed, Predicted, Glassdoor).
    
    Parameters:
        df_name (pd.DataFrame): DataFrame containing observed salary data.
        model: Model for predicting salary.
        encoded_features: Encoded features for prediction.
        specific_cases (list of dict): List of specific cases to analyze, with keys 'job_category', 'seniority_level', and 'country'.
        glassdoor_data (pd.DataFrame): DataFrame with Glassdoor salary range data.
    
    Returns:
        None
    """
    # Initialize lists to store data
    case_labels = []
    observed_percentiles = []
    predicted_percentiles = []
    glassdoor_percentiles = []

    for specific_case in specific_cases:
        job_category = specific_case['job_category']
        seniority_level = specific_case['seniority_level']
        country = specific_case['country']

        # --- Observed Data ---
        salary_data = df_name['salary'][
            (df_name['seniority_level'] == seniority_level) &
            (df_name['job_category'] == job_category) &
            (df_name['country'] == country)
        ]

        if len(salary_data) > 0:
            P25_obs = np.percentile(salary_data, 25)
            P50_obs = np.percentile(salary_data, 50)
            P75_obs = np.percentile(salary_data, 75)
        else:
            P25_obs = P50_obs = P75_obs = np.nan

        observed_percentiles.append([P25_obs, P50_obs, P75_obs])

        # --- Predicted Data ---
        predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(
            model, encoded_features, specific_case
        )

        P25_pred, P50_pred, P75_pred = predicted_salary_p25, predicted_salary, predicted_salary_p75
        predicted_percentiles.append([P25_pred, P50_pred, P75_pred])

        # --- Glassdoor Data ---
        glassdoor_row = glassdoor_data[
            (glassdoor_data['job_category'] == job_category) &
            (glassdoor_data['seniority_level'] == seniority_level) &
            (glassdoor_data['country'] == country)
        ]

        if not glassdoor_row.empty:
            glassdoor_lower = glassdoor_row['glassdoor_lower'].values[0]
            glassdoor_upper = glassdoor_row['glassdoor_upper'].values[0]
            glassdoor_median = (glassdoor_lower + glassdoor_upper) / 2
            P25_glassdoor, P50_glassdoor, P75_glassdoor = glassdoor_lower, glassdoor_median, glassdoor_upper
        else:
            P25_glassdoor = P50_glassdoor = P75_glassdoor = np.nan

        glassdoor_percentiles.append([P25_glassdoor, P50_glassdoor, P75_glassdoor])

        # Add case label
        case_label = f"{seniority_level.capitalize()} + {job_category} + {country.upper()}"
        case_labels.append(case_label)

    # Create DataFrame for plotting
    data_list = []
    for idx, case_label in enumerate(case_labels):
        data_list.append({'Case': case_label, 'Source': 'Observed', 'P25': observed_percentiles[idx][0], 'P50': observed_percentiles[idx][1], 'P75': observed_percentiles[idx][2]})
        data_list.append({'Case': case_label, 'Source': 'Predicted', 'P25': predicted_percentiles[idx][0], 'P50': predicted_percentiles[idx][1], 'P75': predicted_percentiles[idx][2]})
        data_list.append({'Case': case_label, 'Source': 'Glassdoor', 'P25': glassdoor_percentiles[idx][0], 'P50': glassdoor_percentiles[idx][1], 'P75': glassdoor_percentiles[idx][2]})
    
    df_plot = pd.DataFrame(data_list)
    df_plot.sort_values(['Case', 'Source'], inplace=True)

    # Plotting
    cases = df_plot['Case'].unique()
    sources = ['Observed', 'Predicted', 'Glassdoor']
    n_cases = len(cases)
    n_sources = len(sources)
    bar_width = 0.2
    x = np.arange(n_cases)
    offsets = np.linspace(-bar_width * (n_sources - 1) / 2, bar_width * (n_sources - 1) / 2, n_sources)
    source_colors = {'Observed': 'blue', 'Predicted': 'green', 'Glassdoor': 'orange'}

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted_labels = set()

    for i, source in enumerate(sources):
        bar_positions = x + offsets[i]
        medians, lower_errors, upper_errors = [], [], []
        
        for case in cases:
            source_data = df_plot[(df_plot['Case'] == case) & (df_plot['Source'] == source)]
            if not source_data.empty:
                median = source_data['P50'].values[0]
                P25 = source_data['P25'].values[0]
                P75 = source_data['P75'].values[0]
                lower_error = median - P25
                upper_error = P75 - median
            else:
                median = np.nan
                lower_error = upper_error = np.nan
            medians.append(median)
            lower_errors.append(lower_error)
            upper_errors.append(upper_error)
        
        # Convert to numpy arrays and plot
        medians, lower_errors, upper_errors = np.array(medians), np.array(lower_errors), np.array(upper_errors)
        valid = ~np.isnan(medians)
        
        if valid.any():
            if source not in plotted_labels:
                ax.bar(bar_positions[valid], medians[valid], width=bar_width, label=source,
                       color=source_colors.get(source), yerr=[lower_errors[valid], upper_errors[valid]], capsize=5)
                plotted_labels.add(source)
            else:
                ax.bar(bar_positions[valid], medians[valid], width=bar_width,
                       color=source_colors.get(source), yerr=[lower_errors[valid], upper_errors[valid]], capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha='right')
    ax.set_xlabel('Cases')
    ax.set_ylabel('Salary')
    ax.set_title('Salary Percentiles by Case and Source')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ## Combined Dataframe

# ### Initial fitting

# In[105]:


df_name = df_combined
categorical_vars = ['job_category','seniority_level','country']


# In[107]:


# Your prepared DataFrame
X = df_name[categorical_vars]
y = df_name['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())

# Print encoded feature categories (useful for future predictions)
print("\nEncoded feature categories:")
for feature, categories in encoded_features.items():
    print(f"{feature}: {categories}")


# In[109]:


rmse_original_scale = calculate_rmse_original_scale(model, y_aligned)
print("RMSE on Original Scale:", rmse_original_scale)

plot_model_coefficients(model,10,5)


# #### Checking Residuals: Normality

# In[112]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[114]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[115]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[118]:


plot_homoscedasticity(residuals)


# In[119]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=17000, step_size=500, num_simulations=10, dot_size=10)


# ### Influential points

# In[121]:


# Calculate leverage and Cook's Distance using the model
leverage, cooks_d = calculate_leverage_and_cooks_distance(model)

# Plot the leverage and Cook's Distance
plot_leverage_and_cooks_distance(leverage, cooks_d)


# In[123]:


# Leverage OR cook's D EITHER high
# Identify observations with high leverage or high Cook's Distance
n = len(leverage)
p = model.df_model
leverage_threshold = 2 * (p + 1) / n
cooks_threshold = 4 / n

# Identify high leverage and high Cook's Distance points separately
high_leverage_points = np.where(leverage > leverage_threshold)[0]
high_cooks_points = np.where(cooks_d > cooks_threshold)[0]

# Use the union of high leverage and high Cook's Distance points
influential_points = np.union1d(high_leverage_points, high_cooks_points)

print(f"Number of high leverage points: {len(high_leverage_points)}")
print(f"Number of high Cook's Distance points: {len(high_cooks_points)}")
print(f"Number of influential points (either condition): {len(influential_points)}")

# Remove influential observations
X_encoded_cleaned = X_encoded.drop(index=influential_points).reset_index(drop=True)
y_cleaned = y_aligned.drop(index=influential_points).reset_index(drop=True)

# Refit the model without influential observations
model_cleaned = sm.OLS(y_cleaned, X_encoded_cleaned).fit()

# Recalculate leverage and Cook's Distance
leverage_cleaned, cooks_d_cleaned = calculate_leverage_and_cooks_distance(model_cleaned)

# Plot again
plot_leverage_and_cooks_distance(leverage_cleaned, cooks_d_cleaned)


# In[125]:


# Ensure indices are aligned with the original DataFrame
X_encoded_with_index = X_encoded.reset_index(drop=True)
y_aligned_with_index = y_aligned.reset_index(drop=True)

# Original indices (positions after resetting index)
original_indices = X_encoded_with_index.index

# Map influential points to positions in df_combined
# Reset index of df_combined to ensure alignment
df_name_reset = df_name.reset_index(drop=True)

# Ensure that df_combined_reset has the same number of rows as X_encoded_with_index
assert len(df_name_reset) == len(X_encoded_with_index), "Mismatch in number of rows"

# Retrieve influential observations from the original DataFrame
df_name_influential = df_name_reset.iloc[influential_points]

# Remove the influential points from the original data frame, creating a new df
df_name_noninfl = df_name_reset.drop(index=influential_points).reset_index(drop=True)

# Examine influential observations
df_name_influential.head()


# In[126]:


print("Original DataFrame shape:", df_name_reset.shape)
print("Number of influential points:", len(influential_points))
print("DataFrame without influential points shape:", df_name_noninfl.shape)


# #### Comparing VIFs

# In[128]:


# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[129]:


# Define the dataframe to be used
df_name = df_name_noninfl

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ### Refitting w/o influential points

# In[131]:


# Your prepared DataFrame
X = df_name_noninfl[categorical_vars]
y = df_name_noninfl['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())


# In[132]:


rmse_original_scale = calculate_rmse_original_scale(model, y_aligned)
print("RMSE on Original Scale:", rmse_original_scale)

plot_model_coefficients(model,10,3)


# #### Checking Residuals: Normality

# In[135]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[136]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[137]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[139]:


plot_homoscedasticity(residuals)


# In[140]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=17000, step_size=500, num_simulations=10, dot_size=10)


# ### Making a prediction

# In[143]:


specific_case = {
    'job_category': 'Data Scientist/ ML Engineer',
    'seniority_level': 'senior',
    'country': 'us'
}


# In[145]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")


# #### Plotting the prediction

# In[149]:


specific_case = {
    'job_category': 'Data Scientist/ ML Engineer',
    'seniority_level': 'senior',
    'country': 'us'
}


# In[150]:


plot_salary_distributions(df_name, model, encoded_features, specific_case, glassdoor_data)


# ### The result

# In[152]:


# Define your specific cases in the desired order
specific_cases = [
    {'job_category': 'Data Analyst', 'seniority_level': 'junior', 'country': 'de'},
    {'job_category': 'Data Analyst', 'seniority_level': 'medior', 'country': 'de'},
    {'job_category': 'Data Analyst', 'seniority_level': 'senior', 'country': 'de'},
    {'job_category': 'Data Engineer', 'seniority_level': 'junior', 'country': 'de'},
    {'job_category': 'Data Engineer', 'seniority_level': 'medior', 'country': 'de'},
    {'job_category': 'Data Engineer', 'seniority_level': 'senior', 'country': 'de'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'junior', 'country': 'de'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'medior', 'country': 'de'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'senior', 'country': 'de'},
]


# In[154]:


plot_salary_percentiles_by_case(df_name, model, encoded_features, specific_cases, glassdoor_data)


# In[157]:


# Define your specific cases in the desired order
specific_cases = [
    {'job_category': 'Data Analyst', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Analyst', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Analyst', 'seniority_level': 'senior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'senior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'senior', 'country': 'us'},
]


# In[158]:


plot_salary_percentiles_by_case(df_name, model, encoded_features, specific_cases, glassdoor_data)


# ## DF-AI

# ### Initial fitting

# In[164]:


df_name = df_ai_w_l
categorical_vars = ['job_category','seniority_level','country']


# In[165]:


# Your prepared DataFrame
X = df_name[categorical_vars]
y = df_name['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())

# Print encoded feature categories (useful for future predictions)
print("\nEncoded feature categories:")
for feature, categories in encoded_features.items():
    print(f"{feature}: {categories}")


# In[166]:


rmse_original_scale = calculate_rmse_original_scale(model, y_aligned)
print("RMSE on Original Scale:", rmse_original_scale)

plot_model_coefficients(model,10,5)


# #### Checking Residuals: Normality

# In[168]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[170]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[172]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[174]:


plot_homoscedasticity(residuals)


# In[175]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=13000, step_size=500, num_simulations=10, dot_size=10)


# ### Influential points

# In[177]:


# Calculate leverage and Cook's Distance using the model
leverage, cooks_d = calculate_leverage_and_cooks_distance(model)

# Plot the leverage and Cook's Distance
plot_leverage_and_cooks_distance(leverage, cooks_d)


# In[178]:


# Leverage OR cook's D EITHER high
# Identify observations with high leverage or high Cook's Distance
n = len(leverage)
p = model.df_model
leverage_threshold = 2 * (p + 1) / n
cooks_threshold = 4 / n

# Identify high leverage and high Cook's Distance points separately
high_leverage_points = np.where(leverage > leverage_threshold)[0]
high_cooks_points = np.where(cooks_d > cooks_threshold)[0]

# Use the union of high leverage and high Cook's Distance points
influential_points = np.union1d(high_leverage_points, high_cooks_points)

print(f"Number of high leverage points: {len(high_leverage_points)}")
print(f"Number of high Cook's Distance points: {len(high_cooks_points)}")
print(f"Number of influential points (either condition): {len(influential_points)}")

# Remove influential observations
X_encoded_cleaned = X_encoded.drop(index=influential_points).reset_index(drop=True)
y_cleaned = y_aligned.drop(index=influential_points).reset_index(drop=True)

# Refit the model without influential observations
model_cleaned = sm.OLS(y_cleaned, X_encoded_cleaned).fit()

# Recalculate leverage and Cook's Distance
leverage_cleaned, cooks_d_cleaned = calculate_leverage_and_cooks_distance(model_cleaned)

# Plot again
plot_leverage_and_cooks_distance(leverage_cleaned, cooks_d_cleaned)


# In[179]:


# Ensure indices are aligned with the original DataFrame
X_encoded_with_index = X_encoded.reset_index(drop=True)
y_aligned_with_index = y_aligned.reset_index(drop=True)

# Original indices (positions after resetting index)
original_indices = X_encoded_with_index.index

# Map influential points to positions in df_combined
# Reset index of df_combined to ensure alignment
df_name_reset = df_name.reset_index(drop=True)

# Ensure that df_combined_reset has the same number of rows as X_encoded_with_index
assert len(df_name_reset) == len(X_encoded_with_index), "Mismatch in number of rows"

# Retrieve influential observations from the original DataFrame
df_name_influential = df_name_reset.iloc[influential_points]

# Remove the influential points from the original data frame, creating a new df
df_name_noninfl = df_name_reset.drop(index=influential_points).reset_index(drop=True)

# Examine influential observations
df_name_influential.head()


# In[180]:


print("Original DataFrame shape:", df_name_reset.shape)
print("Number of influential points:", len(influential_points))
print("DataFrame without influential points shape:", df_name_noninfl.shape)


# #### Comparing VIFs

# In[184]:


# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[186]:


# Define the dataframe to be used
df_name = df_name_noninfl

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ### Refitting w/o influential points

# In[189]:


# Your prepared DataFrame
X = df_name_noninfl[categorical_vars]
y = df_name_noninfl['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())


# In[192]:


rmse_original_scale = calculate_rmse_original_scale(model, y_aligned)
print("RMSE on Original Scale:", rmse_original_scale)

plot_model_coefficients(model,10,3)


# #### Checking Residuals: Normality

# In[195]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[197]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[199]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[203]:


plot_homoscedasticity(residuals)


# In[206]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=13000, step_size=500, num_simulations=10, dot_size=10)


# ### Making a prediction

# In[209]:


specific_case = {
    'job_category': 'Data Scientist/ ML Engineer',
    'seniority_level': 'senior',
    'country': 'us'
}


# In[210]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")


# #### Plotting the prediction

# In[212]:


specific_case = {
    'job_category': 'Data Scientist/ ML Engineer',
    'seniority_level': 'senior',
    'country': 'us'
}


# In[215]:


plot_salary_distributions(df_name, model, encoded_features, specific_case, glassdoor_data)


# ### The result

# In[217]:


# Define your specific cases in the desired order
specific_cases = [
    {'job_category': 'Data Analyst', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Analyst', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Analyst', 'seniority_level': 'senior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Engineer', 'seniority_level': 'senior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'junior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'medior', 'country': 'us'},
    {'job_category': 'Data Scientist/ ML Engineer', 'seniority_level': 'senior', 'country': 'us'},
]


# In[218]:


plot_salary_percentiles_by_case(df_name, model, encoded_features, specific_cases, glassdoor_data)


# # DRAFT

# # Deeper

# #### Initial run

# In[357]:


df_name = df_it_w_l
categorical_vars = ['job_category','seniority_level','country','language_category','company_size_category','city_category','industry_category']

specific_case = {
    'job_category': 'Data Analyst',
    'seniority_level': 'senior',
    'country': 'de',
    'language_category': 'English-speaking (but not german)',
    'company_size_category': 'l',
    'city_category': 'munich',
    'industry_category': 'manufacturing, transportation, or supply chain'
}


# In[359]:


# Your prepared DataFrame
X = df_name[categorical_vars]
y = df_name['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())


# In[361]:


plot_model_coefficients(model,10,6)


# In[ ]:





# #### Checking Residuals: Normality

# In[365]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[367]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[369]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[372]:


plot_homoscedasticity(residuals)


# In[374]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=4000, step_size=100, num_simulations=10, dot_size=10)


# #### Making an initial prediction

# In[377]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")


# #### Influential points

# In[380]:


# Calculate leverage and Cook's Distance using the model
leverage, cooks_d = calculate_leverage_and_cooks_distance(model)

# Plot the leverage and Cook's Distance
plot_leverage_and_cooks_distance(leverage, cooks_d)


# In[382]:


# Leverage OR cook's D EITHER high
# Identify observations with high leverage or high Cook's Distance
n = len(leverage)
p = model.df_model
leverage_threshold = 2 * (p + 1) / n
cooks_threshold = 4 / n

# Identify high leverage and high Cook's Distance points separately
high_leverage_points = np.where(leverage > leverage_threshold)[0]
high_cooks_points = np.where(cooks_d > cooks_threshold)[0]

# Use the union of high leverage and high Cook's Distance points
influential_points = np.union1d(high_leverage_points, high_cooks_points)

print(f"Number of high leverage points: {len(high_leverage_points)}")
print(f"Number of high Cook's Distance points: {len(high_cooks_points)}")
print(f"Number of influential points (either condition): {len(influential_points)}")

# Remove influential observations
X_encoded_cleaned = X_encoded.drop(index=influential_points).reset_index(drop=True)
y_cleaned = y_aligned.drop(index=influential_points).reset_index(drop=True)

# Refit the model without influential observations
model_cleaned = sm.OLS(y_cleaned, X_encoded_cleaned).fit()

# Recalculate leverage and Cook's Distance
leverage_cleaned, cooks_d_cleaned = calculate_leverage_and_cooks_distance(model_cleaned)

# Plot again
plot_leverage_and_cooks_distance(leverage_cleaned, cooks_d_cleaned)


# In[384]:


# Ensure indices are aligned with the original DataFrame
X_encoded_with_index = X_encoded.reset_index(drop=True)
y_aligned_with_index = y_aligned.reset_index(drop=True)

# Original indices (positions after resetting index)
original_indices = X_encoded_with_index.index

# Map influential points to positions in df_combined
# Reset index of df_combined to ensure alignment
df_name_reset = df_name.reset_index(drop=True)

# Ensure that df_combined_reset has the same number of rows as X_encoded_with_index
assert len(df_name_reset) == len(X_encoded_with_index), "Mismatch in number of rows"

# Retrieve influential observations from the original DataFrame
df_name_influential = df_name_reset.iloc[influential_points]

# Remove the influential points from the original data frame, creating a new df
df_name_noninfl = df_name_reset.drop(index=influential_points).reset_index(drop=True)

# Examine influential observations
#df_name_influential.head()


# In[386]:


print("Original DataFrame shape:", df_name_reset.shape)
print("Number of influential points:", len(influential_points))
print("DataFrame without influential points shape:", df_name_noninfl.shape)


# #### comparing VIFs, Running w/o influential points

# In[389]:


# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[391]:


# Define the dataframe to be used
df_name = df_name_noninfl

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[393]:


# Your prepared DataFrame
X = df_name_noninfl[categorical_vars]
y = df_name_noninfl['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())


# #### Checking Residuals: Normality

# In[396]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[398]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[400]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[403]:


plot_homoscedasticity(residuals)


# In[405]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=3000, step_size=50, num_simulations=10, dot_size=10)


# #### Model interpretation

# In[408]:


plot_model_coefficients(model,10,5)


# #### Making an adjusted prediction

# In[411]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")


# # Deeper # 2

# #### Initial run

# In[505]:


df_name = df_combined
categorical_vars = ['job_category','seniority_level','country','language_category','company_size_category','city_category','industry_category']

specific_case = {
    'job_category': 'Data Analyst',
    'seniority_level': 'senior',
    'country': 'de',
    'language_category': 'English-speaking (but not german)',
    'company_size_category': 'l',
    'city_category': 'munich',
    'industry_category': 'manufacturing, transportation, or supply chain'
}


# In[513]:


# Your prepared DataFrame
X = df_name[categorical_vars]
y = df_name['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)


# View the summary
print(model.summary())


# In[451]:


plot_model_coefficients(model,10,6)


# #### Checking Residuals: Normality

# In[454]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[455]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[456]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[461]:


plot_homoscedasticity(residuals)


# In[462]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=17000, step_size=500, num_simulations=10, dot_size=10)


# #### Making an initial prediction

# In[464]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")


# #### Influential points

# In[468]:


# Calculate leverage and Cook's Distance using the model
leverage, cooks_d = calculate_leverage_and_cooks_distance(model)

# Plot the leverage and Cook's Distance
plot_leverage_and_cooks_distance(leverage, cooks_d)


# In[469]:


# Leverage OR cook's D EITHER high
# Identify observations with high leverage or high Cook's Distance
n = len(leverage)
p = model.df_model
leverage_threshold = 2 * (p + 1) / n
cooks_threshold = 4 / n

# Identify high leverage and high Cook's Distance points separately
high_leverage_points = np.where(leverage > leverage_threshold)[0]
high_cooks_points = np.where(cooks_d > cooks_threshold)[0]

# Use the union of high leverage and high Cook's Distance points
influential_points = np.union1d(high_leverage_points, high_cooks_points)

print(f"Number of high leverage points: {len(high_leverage_points)}")
print(f"Number of high Cook's Distance points: {len(high_cooks_points)}")
print(f"Number of influential points (either condition): {len(influential_points)}")

# Remove influential observations
X_encoded_cleaned = X_encoded.drop(index=influential_points).reset_index(drop=True)
y_cleaned = y_aligned.drop(index=influential_points).reset_index(drop=True)

# Refit the model without influential observations
model_cleaned = sm.OLS(y_cleaned, X_encoded_cleaned).fit()

# Recalculate leverage and Cook's Distance
leverage_cleaned, cooks_d_cleaned = calculate_leverage_and_cooks_distance(model_cleaned)

# Plot again
plot_leverage_and_cooks_distance(leverage_cleaned, cooks_d_cleaned)


# In[470]:


# Ensure indices are aligned with the original DataFrame
X_encoded_with_index = X_encoded.reset_index(drop=True)
y_aligned_with_index = y_aligned.reset_index(drop=True)

# Original indices (positions after resetting index)
original_indices = X_encoded_with_index.index

# Map influential points to positions in df_combined
# Reset index of df_combined to ensure alignment
df_name_reset = df_name.reset_index(drop=True)

# Ensure that df_combined_reset has the same number of rows as X_encoded_with_index
assert len(df_name_reset) == len(X_encoded_with_index), "Mismatch in number of rows"

# Retrieve influential observations from the original DataFrame
df_name_influential = df_name_reset.iloc[influential_points]

# Remove the influential points from the original data frame, creating a new df
df_name_noninfl = df_name_reset.drop(index=influential_points).reset_index(drop=True)

# Examine influential observations
# df_name_influential.head()


# In[472]:


print("Original DataFrame shape:", df_name_reset.shape)
print("Number of influential points:", len(influential_points))
print("DataFrame without influential points shape:", df_name_noninfl.shape)


# #### comparing VIFs, Running w/o influential points

# In[475]:


# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[477]:


# Define the dataframe to be used
df_name = df_name_noninfl

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# In[481]:


# Your prepared DataFrame
X = df_name_noninfl[categorical_vars]
y = df_name_noninfl['salary_log']

# Train the model
model, encoded_features, X_encoded, y_aligned = train_model_1617(X, y)

# View the summary
print(model.summary())


# #### Checking Residuals: Normality

# In[483]:


# Extract residuals from the fitted model
residuals = model.resid
fitted_values = model.fittedvalues


# In[485]:


plot_residuals_diagnostics(residuals, hist_bins=40, qq_marker_size=1, qq_alpha=0.6)


# In[489]:


results_df = simulate_shapiro_test(residuals, min_sample_size=20, max_sample_size=1000, step_size=10, num_simulations=10, dot_size=10)


# #### Checking Residuals: Homoscedasticity

# In[493]:


plot_homoscedasticity(residuals)


# In[495]:


results_df = simulate_breusch_pagan_test(residuals, fitted_values, min_sample_size=20, max_sample_size=17000, step_size=500, num_simulations=10, dot_size=10)


# #### Model interpretation

# In[498]:


plot_model_coefficients(model,10,5)


# #### Making an adjusted prediction

# In[501]:


# Predict salary for the specific case
predicted_salary, predicted_salary_p25, predicted_salary_p75 = predict_salary_for_case_log_perc(model, encoded_features, specific_case)

print(f"Predicted Median Salary: {predicted_salary:.2f}")
print(f"Predicted 25th Percentile Salary: {predicted_salary_p25:.2f}")
print(f"Predicted 75th Percentile Salary: {predicted_salary_p75:.2f}")

