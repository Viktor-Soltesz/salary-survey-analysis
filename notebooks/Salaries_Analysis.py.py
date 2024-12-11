#!/usr/bin/env python
# coding: utf-8

# <font size="6.5"><b>Analyzing Salaries data</b></font>

# <h1 style="background-color: #0e2e3b; color: white; font-size: 40px; font-weight: bold; padding: 10px;"> Import libraries & data, general settings </h1>

# In[4]:


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


# In[5]:


import geopandas as gpd
from shapely.geometry import box


# In[6]:


df_it = pd.read_csv('../data/cleaned/df_it.csv', low_memory=False)
df_k = pd.read_csv('../data/cleaned/df_k.csv', low_memory=False)
df_ai = pd.read_csv('../data/cleaned/df_ai.csv', low_memory=False)


# In[7]:


df_combined_0 = pd.concat([df_ai, df_it, df_k])


# <h2 style="background-color: #07447E; color: white; font-size: 30px; font-weight: bold; padding: 10px;"> Styles </h2>

# In[9]:


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


# In[10]:


print(plt.style.available)


# In[11]:


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


# # Preparing dataframes

# ## Western countries

# In[14]:


firstworld_countries = [
    'al', 'ad', 'am', 'at', 'az', 'by', 'be', 'ba', 'bg', 'hr',
    'cy', 'cz', 'dk', 'ee', 'fi', 'fr', 'ge', 'de', 'gr', 'hu',
    'is', 'ie', 'it', 'kz', 'xk', 'lv', 'li', 'lt', 'lu', 'mt',
    'md', 'mc', 'me', 'nl', 'mk', 'no', 'pl', 'pt', 'ro', 'ru',
    'sm', 'rs', 'sk', 'si', 'es', 'se', 'ch', 'tr', 'ua', 'gb',
    'va', 'ca', 'au', 'us'
]


# In[15]:


#developed_countries = ['de', 'gb', 'nl', 'se', 'dk', 'be', 'fi', 'at', 'ch', 'ie', 'ca', 'au', 'us']
western_countries = ['de', 'at', 'gb', 'fr', 'nl', 'be', 'dk', 'se', 'fi', 'no', 'us', 'ca', 'au']


# ## Exclusion categories

# These are out of scope

# In[18]:


seniority_exclusion = ['other'] #executive

exclude_categories = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers', 'Other managers']
jobcategory_exclusion = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers']


# In[19]:


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


# In[20]:


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


# In[21]:


df_combined = pd.concat([df.dropna(axis=1, how='all') for df in [df_ai_w_l, df_it_w_l, df_k_w_l]])


# In[22]:


print(df_ai_w_l['job_category'].unique())
print(df_it_w_l['job_category'].unique())
print(df_k_w_l['job_category'].unique())


# ## Creating the exclusively data-related dataframes

# In[24]:


data_fields = ['Data Analyst', 'Data Engineer', 'Data Scientist/ ML Engineer']

df_k_data = df_k_w_l[df_k_w_l['job_category'].isin(data_fields)]
df_it_data = df_it_w_l[df_it_w_l['job_category'].isin(data_fields)]
df_ai_data = df_ai_w_l[df_ai_w_l['job_category'].isin(data_fields)]

# df_k_data = df_k_data[(df_k_data['seniority_level'] != 'executive')]
# df_it_data = df_it_data[(df_it_data['seniority_level'] != 'executive')]
# df_it_data = df_it_data[(df_it_data['seniority_level'] != 'other')]
# df_ai_data = df_ai_data[(df_ai_data['seniority_level'] != 'executive')]

# Creating a union from the dataframes
#df_data = pd.concat([df_ai_data, df_k_data, df_it_data], ignore_index=True)
df_data = pd.concat([df.dropna(axis=1, how='all') for df in [df_ai_data, df_k_data, df_it_data]])
df_data_combined = pd.concat([df.dropna(axis=1, how='all') for df in [df_ai_data, df_k_data, df_it_data]])
df_data['year'] = df_data['year'].astype(int)


# #### Listing out the eligible factorial groups for K-S

# In[26]:


# Extract unique (job_category, seniority_level) pairs from each dataframe
groups_ai_factorial = set(df_ai_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
groups_k_factorial = set(df_k_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
groups_it_factorial = set(df_it_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))

groups_ai_factorial_list = sorted(list(groups_ai_factorial))
groups_k_factorial_list = sorted(list(groups_k_factorial))
groups_it_factorial_list = sorted(list(groups_it_factorial))


# In[27]:


print(df_ai_w.groupby(['seniority_level', 'job_category']).size())
print(df_ai_w_l.groupby(['seniority_level', 'job_category']).size())


# In[28]:


# Specify the minimum number of counts required to keep the group
min_count = 20

# Calculate the count of data points for each combination of seniority_level and job_category
group_counts = df_combined.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()

# Rename the count column for clarity
group_counts.rename(columns={'salary_norm': 'count'}, inplace=True)

# Filter groups that meet the minimum count criteria
valid_groups = group_counts[group_counts['count'] >= min_count]

# Merge the valid groups back with the original dataframe to keep only the rows from the valid combinations
df_combined_filtered = pd.merge(df_combined, valid_groups[['seniority_level', 'job_category']], 
                                on=['seniority_level', 'job_category'], 
                                how='inner')


# In[29]:


df_combined_filtered['job_category'].unique()


# ## 'job_categories' that are present in all 3 surveys

# In[31]:


# Extract unique job categories from each dataframe
categories_ai = set(df_ai_w_l['job_category'].unique())
categories_k = set(df_k_w_l['job_category'].unique())
categories_it = set(df_it_w_l['job_category'].unique())

# Find the intersection of all three sets
common_categories = categories_ai & categories_k & categories_it

print("Job categories present in all three surveys:")
common_categories


# ## The common factorial cells

# ### For levene (or ANOVA)

# In[34]:


df_ai_common = df_ai_w_l.copy()
df_k_common = df_k_w_l.copy()
df_it_common = df_it_w_l.copy()

# Extract unique (job_category, seniority_level) pairs from each dataframe
groups_ai = set(df_ai_common[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
groups_k = set(df_k_common[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
groups_it = set(df_it_common[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))

# Find intersection of these sets to get common factorial groups
common_groups = groups_ai & groups_k & groups_it

# Convert the set to a sorted list (optional)
common_groups_list = sorted(list(common_groups))

print("Factorial groups present in all three surveys:")
common_groups_list


# In[35]:


# Filter each dataframe to include only the common factorial groups
df_ai_common = df_ai_common[df_ai_common[['job_category', 'seniority_level']].apply(tuple, axis=1).isin(common_groups)]
df_k_common = df_k_common[df_k_common[['job_category', 'seniority_level']].apply(tuple, axis=1).isin(common_groups)]
df_it_common = df_it_common[df_it_common[['job_category', 'seniority_level']].apply(tuple, axis=1).isin(common_groups)]


# In[36]:


df_common_combined = pd.concat([df.dropna(axis=1, how='all') for df in [df_ai_common, df_k_common, df_it_common]])


# In[37]:


print(len(df_combined_0))
print(len(df_combined))
print(len(df_data))
print(len(df_common_combined))


# # Profile of Survey Respondents

# ## Ratio of the categorical variables

# In[40]:


import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[41]:


# Function to create a treemap figure for a given DataFrame
def create_treemap(df, title):
    counts = df['seniority_level'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = go.Figure(go.Treemap(
        labels=counts['Category'],
        parents=[''] * len(counts),
        values=counts['Count'],
        textinfo="label+value+percent entry",
        hoverinfo="label+value+percent entry"
    ))
    fig.update_layout(title=title, margin=dict(t=50, b=0, l=0, r=0))
    return fig

# Create individual treemaps
treemap_ai = create_treemap(df_ai, 'AI-Jobs.net job-categories distribution')
treemap_bi = create_treemap(df_it, 'DE-IT-Survey job-categories distribution')
treemap_ci = create_treemap(df_k, 'DE-IT-Survey job-categories distribution')

# Create subplots
fig = sp.make_subplots(
    rows=1, cols=3,
    subplot_titles=('AI-Jobs.net', 'DE-IT-Survey', 'Kaggle'),
    specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]  # domain type for treemaps
)

# Add treemaps to subplots
for trace in treemap_ai.data:
    fig.add_trace(trace, row=1, col=1)
for trace in treemap_bi.data:
    fig.add_trace(trace, row=1, col=2)
for trace in treemap_ci.data:
    fig.add_trace(trace, row=1, col=3)

# Update layout
fig.update_layout(
    width=1300,
    height=600,
    title_text="Distribution of Seniority Levels",
    showlegend=False
)

fig.show()


# In[42]:


# Function to create a treemap figure for a given DataFrame
def create_treemap(df, title):
    counts = df['year'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = go.Figure(go.Treemap(
        labels=counts['Category'],
        parents=[''] * len(counts),
        values=counts['Count'],
        textinfo="label+value+percent entry",
        hoverinfo="label+value+percent entry"
    ))
    fig.update_layout(title=title, margin=dict(t=50, b=0, l=0, r=0))
    return fig

# Create individual treemaps
treemap_ai = create_treemap(df_ai, 'AI-Jobs.net job-categories distribution')
treemap_bi = create_treemap(df_it, 'DE-IT-Survey job-categories distribution')
treemap_ci = create_treemap(df_k, 'DE-IT-Survey job-categories distribution')

# Create subplots
fig = sp.make_subplots(
    rows=1, cols=3,
    subplot_titles=('AI-Jobs.net', 'DE-IT-Survey', 'Kaggle'),
    specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]  # domain type for treemaps
)

# Add treemaps to subplots
for trace in treemap_ai.data:
    fig.add_trace(trace, row=1, col=1)
for trace in treemap_bi.data:
    fig.add_trace(trace, row=1, col=2)
for trace in treemap_ci.data:
    fig.add_trace(trace, row=1, col=3)

# Update layout
fig.update_layout(
    width=1300,
    height=600,
    title_text="Responses by year",
    showlegend=False
)

fig.show()


# In[43]:


# Count the number of respondents per seniority level
country_counts = df_ai['country'].value_counts().reset_index()
country_counts.columns = ['Seniority Level', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['Seniority Level'],
                 values='Count',
                 title='Distribution of Country',
                 color='Seniority Level')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# In[44]:


# Count the number of respondents per seniority level
country_counts = df_k['country'].value_counts().reset_index()
country_counts.columns = ['Seniority Level', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['Seniority Level'],
                 values='Count',
                 title='Distribution of Country',
                 color='Seniority Level')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# ## Germany-IT specific

# In[46]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['age'], bins=40, range=(0, 100), edgecolor='black')
axs[0].set_title('Histogram of age')
axs[0].set_xlabel('age')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_it['experience'], bins=40, range=(0, 30), edgecolor='black')
axs[1].set_title('Histogram of experience')
axs[1].set_xlabel('experience')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_it['years_of_experience_in_germany'], bins=40, range=(0, 30), edgecolor='black')
axs[2].set_title('Histogram of Years spent in Germany')
axs[2].set_xlabel('years_of_experience_in_germany')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# In[47]:


# df_k = pd.DataFrame({'education_level': ['Bsc', 'Msc', 'Phd', 'Msc', 'Bsc', 'Bsc', 'Phd', 'Msc']})

# Data to plot
education_counts = df_it['company_size'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of Company Size')
plt.show()


# In[48]:


# Example DataFrame
# df_k = pd.DataFrame({'education_level': ['Bsc', 'Msc', 'Phd', 'Msc', 'Bsc', 'Bsc', 'Phd', 'Msc']})

# Data to plot
education_counts = df_it['language_category'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.85,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of Language used at work')
plt.show()


# In[49]:


# Count the number of respondents per seniority level
country_counts = df_it['language_at_work'].value_counts().reset_index()
country_counts.columns = ['language_at_work', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['language_at_work'],
                 values='Count',
                 title='Distribution language used at work',
                 color='language_at_work')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# In[50]:


# Count the number of respondents per seniority level
country_counts = df_it['skills'].value_counts().reset_index()
country_counts.columns = ['skills', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['skills'],
                 values='Count',
                 title='Distribution of skills',
                 color='skills')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# In[51]:


# Data to plot
education_counts = df_it['company_industry'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.85,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of Industry field')
plt.show()


# In[52]:


# Count the number of respondents per seniority level
country_counts = df_it['company_industry'].value_counts().reset_index()
country_counts.columns = ['company_industry', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['company_industry'],
                 values='Count',
                 title='Distribution of Industry',
                 color='company_industry')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# In[53]:


# Count the number of respondents per seniority level
country_counts = df_it['job_category'].value_counts().reset_index()
country_counts.columns = ['job_category', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['job_category'],
                 values='Count',
                 title='Distribution of Job categories',
                 color='job_category')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# ## Kaggle-specific

# In[55]:


df_k.head()


# In[56]:


df_k['education_level'].unique()


# In[57]:


# df_k = pd.DataFrame({'education_level': ['Bsc', 'Msc', 'Phd', 'Msc', 'Bsc', 'Bsc', 'Phd', 'Msc']})

# Data to plot
education_counts = df_k['education_level'].value_counts()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of Education Levels')
plt.show()


# In[58]:


# Data to plot
education_counts = df_k['company_size'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of company size')
plt.show()


# In[59]:


# Data to plot
education_counts = df_k['industry'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.85,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of industry field')
plt.show()


# In[60]:


# Count the number of respondents per seniority level
country_counts = df_k['job_category'].value_counts().reset_index()
country_counts.columns = ['job_category', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['job_category'],
                 values='Count',
                 title='Distribution of job category',
                 color='job_category')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# ## AI-Jobs.net

# In[62]:


df_ai.head()


# In[63]:


# Count the number of respondents per seniority level
country_counts = df_ai['job_category'].value_counts().reset_index()
country_counts.columns = ['job_category', 'Count']

# Create the treemap
fig = px.treemap(country_counts,
                 path=['job_category'],
                 values='Count',
                 title='Distribution of job category',
                 color='job_category')
                 #color_discrete_map=custom_colors)

fig.update_layout(width=1200, height=600)

fig.show()


# In[64]:


# Data to plot
education_counts = df_ai['company_size'].value_counts().sort_values()
labels = education_counts.index
sizes = education_counts.values

# Create an explode list to separate slices for better readability
explode = [0.1] * len(labels)  # Explode all slices for visibility

# Create the ring plot (donut chart)
plt.figure(figsize=(4, 4))
wedges, texts, autotexts = plt.pie(
    sizes,
    #explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    #shadow=True,
    startangle=0,
    pctdistance=0.8,  # Distance of percentage from the center
    labeldistance=1.1, # Distance of labels from the center
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1.5)  # This creates the ring effect
)

# Formatting labels and percentages
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')

plt.title('Distribution of company size')
plt.show()


# In[65]:


# Function to create a treemap figure for a given DataFrame
def create_treemap(df, title):
    counts = df['job_category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = go.Figure(go.Treemap(
        labels=counts['Category'],
        parents=[''] * len(counts),
        values=counts['Count'],
        textinfo="label+value+percent entry",
        hoverinfo="label+value+percent entry"
    ))
    fig.update_layout(title=title, margin=dict(t=50, b=0, l=0, r=0))
    return fig

# Create individual treemaps
treemap_ai = create_treemap(df_ai, 'AI-Jobs.net job-categories distribution')
treemap_bi = create_treemap(df_it, 'DE-IT-Survey job-categories distribution')
treemap_ci = create_treemap(df_k, 'Kaggle job-categories distribution')

# Create subplots
fig = sp.make_subplots(
    rows=1, cols=3,
    subplot_titles=('AI Jobs Distribution', 'BI Jobs Distribution', 'CI Jobs Distribution'),
    specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]  # domain type for treemaps
)

# Add treemaps to subplots
for trace in treemap_ai.data:
    fig.add_trace(trace, row=1, col=1)
for trace in treemap_bi.data:
    fig.add_trace(trace, row=1, col=2)
for trace in treemap_ci.data:
    fig.add_trace(trace, row=1, col=3)

# Update layout
fig.update_layout(
    width=1300,
    height=600,
    title_text="Distribution of Job Categories",
    showlegend=False
)

fig.show()


# # Strength of Association analysis

# In[67]:


from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[68]:


# Function to calculate Cramér's V and its confidence interval
def cramers_v_ci(x, y, alpha=0.05):
    confusion_matrix = pd.crosstab(x, y)
    chi2_stat = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    cramers_v_value = np.sqrt(chi2_stat / (n * (min(r, k) - 1)))
    
    # Calculate confidence interval for chi-squared statistic
    df = (r - 1) * (k - 1)  # degrees of freedom
    chi2_lower = chi2.ppf(alpha / 2, df)
    chi2_upper = chi2.ppf(1 - alpha / 2, df)
    
    # Convert chi-squared CIs to Cramér's V CIs
    lower_v = np.sqrt(chi2_lower / (n * (min(r, k) - 1)))
    upper_v = np.sqrt(chi2_upper / (n * (min(r, k) - 1)))
    
    return cramers_v_value, lower_v, upper_v


# ## Ai-jobs.

# In[70]:


categorical_columns = ['seniority_level', 'country', 'job_category', 'year', 'company_size','company_location']
df_name = df_ai.copy()


# In[71]:


# Create an empty DataFrame to store Chi-squared p-values
chi2_p_matrix = pd.DataFrame(np.ones((len(categorical_columns), len(categorical_columns))), 
                             index=categorical_columns, columns=categorical_columns)

# Calculate the Chi-squared test p-value for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            confusion_matrix = pd.crosstab(df_name[col1], df_name[col2])
            chi2_var, p, _, _ = chi2_contingency(confusion_matrix)
            chi2_p_matrix.loc[col1, col2] = p
        else:
            chi2_p_matrix.loc[col1, col2] = 0.0  # No p-value for comparison with itself

# Round the p-values to three decimal places for better readability
chi2_p_matrix_rounded = chi2_p_matrix.round(3)

# Plotting the heatmap with rounded p-values
plt.figure(figsize=(4, 3))
sns.heatmap(chi2_p_matrix_rounded, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title("Chi-squared Test P-values Heatmap")
plt.show()


# The p-values in this heatmap indicate whether there is a statistically significant association between each pair of categorical variables.
# A p-value < 0.05 (common threshold) suggests a statistically significant association, meaning the relationship is unlikely to have occurred by chance.

# In[73]:


# Create an empty DataFrame to store Cramér's V values and confidence intervals
cramers_v_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                                index=categorical_columns, columns=categorical_columns)
ci_matrix_lower = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)
ci_matrix_upper = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)

# Calculate Cramér's V and confidence intervals for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            v, lower_v, upper_v = cramers_v_ci(df_name[col1], df_name[col2])
            cramers_v_matrix.loc[col1, col2] = v
            ci_matrix_lower.loc[col1, col2] = lower_v
            ci_matrix_upper.loc[col1, col2] = upper_v
        else:
            cramers_v_matrix.loc[col1, col2] = 1.0  # Perfect correlation with itself
            ci_matrix_lower.loc[col1, col2] = 1.0
            ci_matrix_upper.loc[col1, col2] = 1.0

# Combine Cramér's V values and confidence intervals into one annotation format
annot_matrix = cramers_v_matrix.round(2).astype(str) + "\n[-" + ci_matrix_lower.round(2).astype(str) + ", +" + ci_matrix_upper.round(2).astype(str) + "]"

# Plotting the heatmap with annotations
plt.figure(figsize=(12, 4))
sns.heatmap(cramers_v_matrix, annot=annot_matrix, fmt='', cmap='YlGnBu', vmin=0, vmax=1, cbar_kws={'label': "Cramér's V"})
plt.title("Cramér's V Heatmap with Confidence Intervals")
plt.show()


# ## DE IT-survey

# In[75]:


categorical_columns = ['seniority_level', 'job_category', 'year','experience', 'company_type', 'company_size', 'language_at_work', 'years_of_experience_in_germany']
df_name = df_it.copy()


# In[76]:


# Create an empty DataFrame to store Chi-squared p-values
chi2_p_matrix = pd.DataFrame(np.ones((len(categorical_columns), len(categorical_columns))), 
                             index=categorical_columns, columns=categorical_columns)

# Calculate the Chi-squared test p-value for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            confusion_matrix = pd.crosstab(df_name[col1], df_name[col2])
            chi2_var, p, _, _ = chi2_contingency(confusion_matrix)
            chi2_p_matrix.loc[col1, col2] = p
        else:
            chi2_p_matrix.loc[col1, col2] = 0.0  # No p-value for comparison with itself

# Round the p-values to three decimal places for better readability
chi2_p_matrix_rounded = chi2_p_matrix.round(3)

# Plotting the heatmap with rounded p-values
plt.figure(figsize=(4, 3))
sns.heatmap(chi2_p_matrix_rounded, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title("Chi-squared Test P-values Heatmap")
plt.show()


# The p-values in this heatmap indicate whether there is a statistically significant association between each pair of categorical variables.
# A p-value < 0.05 (common threshold) suggests a statistically significant association, meaning the relationship is unlikely to have occurred by chance.

# In[78]:


# Create an empty DataFrame to store Cramér's V values and confidence intervals
cramers_v_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                                index=categorical_columns, columns=categorical_columns)
ci_matrix_lower = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)
ci_matrix_upper = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)

# Calculate Cramér's V and confidence intervals for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            v, lower_v, upper_v = cramers_v_ci(df_name[col1], df_name[col2])
            cramers_v_matrix.loc[col1, col2] = v
            ci_matrix_lower.loc[col1, col2] = lower_v
            ci_matrix_upper.loc[col1, col2] = upper_v
        else:
            cramers_v_matrix.loc[col1, col2] = 1.0  # Perfect correlation with itself
            ci_matrix_lower.loc[col1, col2] = 1.0
            ci_matrix_upper.loc[col1, col2] = 1.0

# Combine Cramér's V values and confidence intervals into one annotation format
annot_matrix = cramers_v_matrix.round(2).astype(str) + "\n[-" + ci_matrix_lower.round(2).astype(str) + "/ +" + ci_matrix_upper.round(2).astype(str) + "]"

# Plotting the heatmap with annotations
plt.figure(figsize=(12, 4))
sns.heatmap(cramers_v_matrix, annot=annot_matrix, fmt='', cmap='YlGnBu', vmin=0, vmax=1, cbar_kws={'label': "Cramér's V"})
plt.title("Cramér's V Heatmap with Confidence Intervals")
plt.show()


# ## Kaggle

# In[80]:


categorical_columns = ['seniority_level', 'country', 'job_category', 'year','education_level', 'company_size']
df_name = df_k.copy()


# In[81]:


# Create an empty DataFrame to store Chi-squared p-values
chi2_p_matrix = pd.DataFrame(np.ones((len(categorical_columns), len(categorical_columns))), 
                             index=categorical_columns, columns=categorical_columns)

# Calculate the Chi-squared test p-value for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            confusion_matrix = pd.crosstab(df_name[col1], df_name[col2])
            chi2_var, p, _, _ = chi2_contingency(confusion_matrix)
            chi2_p_matrix.loc[col1, col2] = p
        else:
            chi2_p_matrix.loc[col1, col2] = 0.0  # No p-value for comparison with itself

# Round the p-values to three decimal places for better readability
chi2_p_matrix_rounded = chi2_p_matrix.round(3)

# Plotting the heatmap with rounded p-values
plt.figure(figsize=(4, 3))
sns.heatmap(chi2_p_matrix_rounded, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title("Chi-squared Test P-values Heatmap")
plt.show()


# The p-values in this heatmap indicate whether there is a statistically significant association between each pair of categorical variables.
# A p-value < 0.05 (common threshold) suggests a statistically significant association, meaning the relationship is unlikely to have occurred by chance.

# In[83]:


# Encode categorical variables using .cat.codes
for column in ['seniority_level', 'country', 'job_category', 'year','education_level', 'company_size']:
    df_name[column] = df_name[column].astype('category').cat.codes


# In[84]:


# Create an empty DataFrame to store Cramér's V values and confidence intervals
cramers_v_matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                                index=categorical_columns, columns=categorical_columns)
ci_matrix_lower = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)
ci_matrix_upper = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), 
                               index=categorical_columns, columns=categorical_columns)

# Calculate Cramér's V and confidence intervals for each pair of categorical variables
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            v, lower_v, upper_v = cramers_v_ci(df_name[col1], df_name[col2])
            cramers_v_matrix.loc[col1, col2] = v
            ci_matrix_lower.loc[col1, col2] = lower_v
            ci_matrix_upper.loc[col1, col2] = upper_v
        else:
            cramers_v_matrix.loc[col1, col2] = 1.0  # Perfect correlation with itself
            ci_matrix_lower.loc[col1, col2] = 1.0
            ci_matrix_upper.loc[col1, col2] = 1.0

# Combine Cramér's V values and confidence intervals into one annotation format
annot_matrix = cramers_v_matrix.round(2).astype(str) + "\n[-" + ci_matrix_lower.round(2).astype(str) + ", +" + ci_matrix_upper.round(2).astype(str) + "]"

# Plotting the heatmap with annotations
plt.figure(figsize=(12, 4))
sns.heatmap(cramers_v_matrix, annot=annot_matrix, fmt='', cmap='YlGnBu', vmin=0, vmax=1, cbar_kws={'label': "Cramér's V"})
plt.title("Cramér's V Heatmap with Confidence Intervals")
plt.show()


# In[85]:


df_k.head()


# # Visualizing

# ## Parallel Coordinates Plot

# In[88]:


# Step 1: Convert categorical variables into numerical codes for better plotting
df_combined_encoded = df_combined.copy()

# Encode categorical variables using .cat.codes
for column in ['survey', 'job_category', 'seniority_level', 'country', 'year']:
    df_combined_encoded[column] = df_combined_encoded[column].astype('category').cat.codes

# Step 2: Create a parallel coordinates plot using Plotly
fig = px.parallel_coordinates(df_combined_encoded,
                              dimensions=['job_category', 'seniority_level', 'country', 'year', 'survey'],
                              color='survey',
                              #opcaity = 0.5,# Color by the 'survey' column
                              color_continuous_scale=px.colors.sequential.Viridis,  # Use a color scale
                              labels={"job_category": "Job Category",
                                      "seniority_level": "Seniority Level",
                                      "country": "Country",
                                      "year": "Year",
                                      "survey": "Survey"},
                              title="Parallel Coordinates Plot of Software Developer Survey Data")

# Step 3: Update layout for better visual appeal
fig.update_layout(
    title_font_size=18,
    title_x=0.5,  # Center the title
    coloraxis_colorbar=dict(
        title="Survey"
    )
)

# Show the plot
fig.show()


# ## Histograms

# In[90]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['salary'], bins=20, range=(0, 300000), edgecolor='black')
axs[0].set_title('Salary Distribution in Germany-IT')
axs[0].set_xlabel('Salary')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_ai['salary'], bins=20, range=(0, 300000), edgecolor='black')
axs[1].set_title('Salary Distribution in AI-Jobs.net')
axs[1].set_xlabel('Salary')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_k['salary'], bins=20, range=(0, 300000), edgecolor='black')
axs[2].set_title('Salary Distribution in Kaggle')
axs[2].set_xlabel('Salary')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# In[91]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['salary'][df_it['country']=='de'], bins=20, range=(0, 300000), edgecolor='black')
axs[0].set_title('Salary Distribution in Germany-IT')
axs[0].set_xlabel('Salary')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_ai['salary'][df_ai['country']=='de'], bins=20, range=(0, 300000), edgecolor='black')
axs[1].set_title('Salary Distribution in AI-Jobs.net')
axs[1].set_xlabel('Salary')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_k['salary'][df_k['country']=='de'], bins=20, range=(0, 300000), edgecolor='black')
axs[2].set_title('Salary Distribution in Kaggle')
axs[2].set_xlabel('Salary')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# In[92]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['salary'][(df_it['country']=='de') & (df_it['job_category']=='Data Analyst')], bins=20, range=(0, 300000), edgecolor='black')
axs[0].set_title('Salary Distribution in Germany-IT')
axs[0].set_xlabel('Salary')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_ai['salary'][(df_ai['country']=='de') & (df_ai['job_category']=='Data Analyst')], bins=20, range=(0, 300000), edgecolor='black')
axs[1].set_title('Salary Distribution in AI-Jobs.net')
axs[1].set_xlabel('Salary')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_k['salary'][(df_k['country']=='de') & (df_k['job_category']=='Data Analyst')], bins=20, range=(0, 300000), edgecolor='black')
axs[2].set_title('Salary Distribution in Kaggle')
axs[2].set_xlabel('Salary')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# In[93]:


df_k.head(2)


# In[94]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['salary_norm'][(df_it['seniority_level']=='senior') & (df_it['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[0].set_title('Salary Distribution in Germany-IT')
axs[0].set_xlabel('Salary')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_ai['salary_norm'][(df_ai['seniority_level']=='senior') & (df_ai['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[1].set_title('Salary Distribution in AI-Jobs.net')
axs[1].set_xlabel('Salary')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_k['salary_norm'][(df_k['seniority_level']=='senior') & (df_k['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[2].set_title('Salary Distribution in Kaggle')
axs[2].set_xlabel('Salary')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# In[95]:


fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=False)

# Plot histogram for the 'salary' column from the first DataFrame
axs[0].hist(df_it['salary_norm_2024'][(df_it['seniority_level']=='senior') & (df_it['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[0].set_title('Salary Distribution in Germany-IT')
axs[0].set_xlabel('Salary')
axs[0].set_ylabel('Frequency')

# Plot histogram for the 'salary' column from the second DataFrame
axs[1].hist(df_ai['salary_norm_2024'][(df_ai['seniority_level']=='senior') & (df_ai['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[1].set_title('Salary Distribution in AI-Jobs.net')
axs[1].set_xlabel('Salary')

# Plot histogram for the 'salary' column from the third DataFrame
axs[2].hist(df_k['salary_norm_2024'][(df_k['seniority_level']=='senior') & (df_k['job_category']=='Data Analyst')], bins=20, range=(0, 4), edgecolor='black')
axs[2].set_title('Salary Distribution in Kaggle')
axs[2].set_xlabel('Salary')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# Conclusion:
# 
# The distributions are often skewed to the right. This is expected of salary data.
# For measuring central tendency, Median is reasonable.
# 
# The more variable I control for, the more similar the ditributions become. This is expected, and substantiates a multivariate analysis that controls for multiple variables.

# ## Yearly trends

# In[98]:


# Real U.S. inflation rates from 2018 to 2024
us_inflation_rates = [1.024, 1.018, 1.012, 1.047, 1.070, 1.035, 1.030]  # Approximate U.S. inflation rates

# Real Germany inflation rates from 2018 to 2024
germany_inflation_rates = [1.019, 1.014, 1.005, 1.031, 1.069, 1.045, 1.030]  # Approximate Germany inflation rates

# Create inflation values starting from 1.0 in 2018 for both countries
inflation_values_us = [1.0]  # Start at 1.0 in 2018 for the U.S.
inflation_values_germany = [1.0]  # Start at 1.0 in 2018 for Germany

for rate_us, rate_germany in zip(us_inflation_rates, germany_inflation_rates):
    inflation_values_us.append(inflation_values_us[-1] * rate_us)
    inflation_values_germany.append(inflation_values_germany[-1] * rate_germany)

# Create DataFrames for U.S. and Germany inflation values
years = np.arange(2018, 2025)  # Years from 2018 to 2024 inclusive
df_inflation_us = pd.DataFrame({
    'year': years,
    'inflation_factor_us': inflation_values_us[:-1]  # Remove the last extrapolation
})

df_inflation_germany = pd.DataFrame({
    'year': years,
    'inflation_factor_germany': inflation_values_germany[:-1]  # Remove the last extrapolation
})

# Plot the normalized salary data
sns.lineplot(data=df_data, x='year', y='salary_norm', hue='survey', marker='o')

# Plot the U.S. recorded inflation line
plt.plot(df_inflation_us['year'], df_inflation_us['inflation_factor_us'], color='magenta', linestyle='--', marker='o', label='U.S. Recorded Inflation')
# Plot the Germany recorded inflation line
plt.plot(df_inflation_germany['year'], df_inflation_germany['inflation_factor_germany'], color='purple', linestyle='--', marker='o', label='Germany Recorded Inflation')

# Customize plot appearance
plt.title('Average Salary Change Through the Years (With U.S. and Germany Recorded Inflation)')
plt.xlabel('Year')
plt.ylabel('Average Salary (Normalized)')
plt.ylim(1, 3)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


# In[99]:


# Real U.S. inflation rates from 2018 to 2024
us_inflation_rates = [1.024, 1.018, 1.012, 1.047, 1.070, 1.035, 1.030]  # Approximate U.S. inflation rates

# Real Germany inflation rates from 2018 to 2024
germany_inflation_rates = [1.019, 1.014, 1.005, 1.031, 1.069, 1.045, 1.030]  # Approximate Germany inflation rates

# Create inflation values starting from 1.0 in 2018 for both countries
inflation_values_us = [1.0]  # Start at 1.0 in 2018 for the U.S.
inflation_values_germany = [1.0]  # Start at 1.0 in 2018 for Germany

for rate_us, rate_germany in zip(us_inflation_rates, germany_inflation_rates):
    inflation_values_us.append(inflation_values_us[-1] * rate_us)
    inflation_values_germany.append(inflation_values_germany[-1] * rate_germany)

# Create DataFrames for U.S. and Germany inflation values
years = np.arange(2018, 2025)  # Years from 2018 to 2024 inclusive
df_inflation_us = pd.DataFrame({
    'year': years,
    'inflation_factor_us': inflation_values_us[:-1]  # Remove the last extrapolation
})

df_inflation_germany = pd.DataFrame({
    'year': years,
    'inflation_factor_germany': inflation_values_germany[:-1]  # Remove the last extrapolation
})

# Plot the normalized salary data
sns.lineplot(data=df_data, x='year', y='salary_norm_2024', hue='survey', marker='o')

# Plot the U.S. recorded inflation line
plt.plot(df_inflation_us['year'], df_inflation_us['inflation_factor_us'], color='magenta', linestyle='--', marker='o', label='U.S. Recorded Inflation')
# Plot the Germany recorded inflation line
plt.plot(df_inflation_germany['year'], df_inflation_germany['inflation_factor_germany'], color='purple', linestyle='--', marker='o', label='Germany Recorded Inflation')

# Customize plot appearance
plt.title('Average Salary Change Through the Years (With U.S. and Germany Recorded Inflation)')
plt.xlabel('Year')
plt.ylabel('Average Salary (Normalized)')
plt.ylim(1, 3)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


# In[100]:


# Compute median salary, Q1, and Q3 for each combination of year and survey
summary_stats = df_data.groupby(['year', 'survey'])['salary_norm'].describe(percentiles=[.25, .75]).reset_index()

# Rename columns for clarity
summary_stats = summary_stats.rename(columns={'25%': 'Q1', '75%': 'Q3'})

# Compute IQR
summary_stats['IQR'] = summary_stats['Q3'] - summary_stats['Q1']

# Plot the median salary with IQR as shaded region
plt.figure(figsize=(10, 6))
for survey in summary_stats['survey'].unique():
    survey_data = summary_stats[summary_stats['survey'] == survey]
    plt.plot(survey_data['year'], survey_data['50%'], marker='o', label=f'{survey} Median')
    plt.fill_between(survey_data['year'], survey_data['Q1'], survey_data['Q3'], alpha=0.3)

plt.title('Median Salary Change Through the Years with IQR')
plt.xlabel('Year')
plt.ylabel('Median Salary')
plt.ylim(1, 3)
plt.grid(True)
plt.legend(title='Survey')
plt.show()


# In[101]:


df_data_plot = df_data.copy()
df_data_plot = df_data_plot[df_data_plot['country'].isin(western_countries)]
#df_data_plot = df_data
df_data_plot = df_data_plot[df_data_plot['seniority_level'] == 'senior']
#df_data_plot = df_data_plot[~((df_data_plot['survey'] == 'ai') & (df_data_plot['year'] == 2020))]

sns.lineplot(data=df_data_plot, x='year', y='salary_norm', hue='job_category', marker='o')

plt.title('Average Salary Change Through the Years')
plt.xlabel('Year')
plt.ylabel('Average Salary')
plt.ylim(1, 3)
plt.grid(True)

plt.show()


# In[102]:


df_data_plot = df_data.copy()
df_data_plot = df_data_plot[df_data_plot['country'].isin(western_countries)]
#df_data_plot = df_data
df_data_plot = df_data_plot[df_data_plot['seniority_level'] == 'senior']
#df_data_plot = df_data_plot[~((df_data_plot['survey'] == 'ai') & (df_data_plot['year'] == 2020))]

sns.lineplot(data=df_data_plot, x='year', y='salary_norm_2024', hue='job_category', marker='o')

plt.title('Average Salary Change Through the Years')
plt.xlabel('Year')
plt.ylabel('Average Salary')
plt.ylim(1, 3)
plt.grid(True)

plt.show()


# In[103]:


# Filter the data for Western countries
df_data_plot = df_data.copy()
df_data_plot = df_data_plot[df_data_plot['country'].isin(western_countries)]

# Group by 'year' and 'seniority_level' to compute the median of 'salary_norm'
df_medians = df_data_plot.groupby(['year', 'seniority_level'], as_index=False)['salary_norm'].median()

# Plot using seaborn lineplot
sns.lineplot(data=df_medians, x='year', y='salary_norm', hue='seniority_level', marker='o')

# Set plot titles and labels
plt.title('Median Salary Change Through the Years')
plt.xlabel('Year')
plt.ylabel('Median Salary')
plt.ylim(1, 3)
plt.grid(True)

# Show the plot
plt.show()


# In[104]:


# Filter the data for Western countries
df_data_plot = df_data.copy()
df_data_plot = df_data_plot[df_data_plot['country'].isin(western_countries)]

# Group by 'year' and 'seniority_level' to compute the median of 'salary_norm'
df_medians = df_data_plot.groupby(['year', 'seniority_level'], as_index=False)['salary_norm_2024'].median()

# Plot using seaborn lineplot
sns.lineplot(data=df_medians, x='year', y='salary_norm_2024', hue='seniority_level', marker='o')

# Set plot titles and labels
plt.title('Median Salary Change Through the Years')
plt.xlabel('Year')
plt.ylabel('Median Salary')
plt.ylim(1, 3)
plt.grid(True)

# Show the plot
plt.show()


# ## Map

# In[106]:


import geopandas as gpd
from shapely.geometry import box

# Load the shapefile or GeoDataFrame (use appropriate path)
shapefile_path = '../data/world_shapefile/ne_110m_admin_0_countries.shp'
world = gpd.read_file(shapefile_path)


# ### World

# In[108]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

# Calculate average salary
median_salary = df_data_seniors.groupby('country')['salary'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
world.boundary.plot(ax=ax)
world.plot(column='salary', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.4},  # Adjust the size of the legend
           cmap='OrRd')  # Choose a colormap

plt.show()


# In[109]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

# Calculate average salary
median_salary = df_data_seniors.groupby('country')['salary_norm_2024'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
world.boundary.plot(ax=ax)
world.plot(column='salary_norm_2024', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.4},  # Adjust the size of the legend
           cmap='OrRd')  # Choose a colormap

plt.show()


# In[110]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

# Calculate average salary
median_salary = df_data_seniors.groupby('country')['salary_normse_2024'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
world.boundary.plot(ax=ax)
world.plot(column='salary_normse_2024', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.4},  # Adjust the size of the legend
           cmap='OrRd')  # Choose a colormap

plt.show()


# ### Europe

# In[112]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

# Calculate median salary instead of mean
median_salary = df_data_seniors.groupby('country')['salary'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Define the bounding box for Europe (minx, miny, maxx, maxy)
bbox = (-30, 35, 60, 72)

# Create a polygon for the bounding box
bbox_polygon = box(*bbox)

# Clip the GeoDataFrame to the bounding box polygon
world = world.clip(bbox_polygon)

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
world.boundary.plot(ax=ax)
world.plot(column='salary', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.6},
           cmap='OrRd')  # Choose a colormap

plt.show()


# In[113]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

#firstworld_countries

# Calculate median salary instead of mean
median_salary = df_data_seniors.groupby('country')['salary_norm_2024'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Define the bounding box for Europe (minx, miny, maxx, maxy)
bbox = (-30, 35, 60, 72)

# Create a polygon for the bounding box
bbox_polygon = box(*bbox)

# Clip the GeoDataFrame to the bounding box polygon
world = world.clip(bbox_polygon)

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
world.boundary.plot(ax=ax)
world.plot(column='salary_norm_2024', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.6},
           cmap='OrRd')  # Choose a colormap

plt.show()


# In[114]:


df_data_seniors = df_combined_0.copy()
df_data_seniors = df_data_seniors[df_data_seniors['seniority_level'] == 'senior']
df_data_seniors['country'] = df_data_seniors['country'].str.upper()

#firstworld_countries

# Calculate median salary instead of mean
median_salary = df_data_seniors.groupby('country')['salary_normmed_2024'].median().reset_index()

# Merge DataFrame with GeoDataFrame
world = gpd.read_file(shapefile_path)
world = world.merge(median_salary, left_on='ISO_A2_EH', right_on='country', how='left')

# Define the bounding box for Europe (minx, miny, maxx, maxy)
bbox = (-30, 35, 60, 72)

# Create a polygon for the bounding box
bbox_polygon = box(*bbox)

# Clip the GeoDataFrame to the bounding box polygon
world = world.clip(bbox_polygon)

# Plot the heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
world.boundary.plot(ax=ax)
world.plot(column='salary_normmed_2024', ax=ax, legend=True,
           legend_kwds={'label': "Median Salary by Country",
                        'orientation': "vertical",
                        'shrink': 0.6},
           cmap='OrRd')  # Choose a colormap

plt.show()


# In[115]:


# Filter data for seniors and the specified countries
df_data_seniors = df_data[df_data['seniority_level'] == 'senior'].copy()
#df_data_seniors = df_data_seniors[df_data_seniors['survey'] != 'k']
df_data_seniors = df_data_seniors[df_data_seniors['country'].isin(western_countries)]

# Calculate the median salary for each country and sort by this value
median_salary = df_data_seniors.groupby('country')['salary_norm_2024'].median().sort_values(ascending=False).index

# Reorder the 'country' column based on the sorted median salary
df_data_seniors['group'] = pd.Categorical(df_data_seniors['country'], categories=median_salary, ordered=True)

# Plot the boxplot with reordered groups
plt.figure(figsize=(15, 5))
sns.boxplot(x='group', y='salary_norm_2024', data=df_data_seniors)
plt.xticks(rotation=90)
plt.ylim(0, 7.5)
plt.title('Salary Distribution by Groups (Reordered by Median Salary)')
plt.show()


# \
# \
# Conclusion:
# - Country seems to be an enormous factor.
# - In the western world, normalizing salary by GDP-Per-Capita seems to be a good step (considering that one of the Surveys was exclusively been made from Germany.)
# - Furthermire, by normalizing, we can avoid grouping everything by by country.
# - Ukraine seems to be an outlier, but that can be caused by dislocated citizens by the Ukranian war, across Europe.
# \
# \
# Therefore, I use normalized salaries in the Western World.

# ## Barplot

# ### Preparation for barplots

# In[119]:


# # Filters for: western countries, and specific categories
# df_ai_w= df_ai.copy()
# df_it_w= df_it.copy()
# df_k_w = df_k.copy()
# 
# # Western countries
# df_ai_w = df_ai_w[df_ai_w['country'].isin(western_countries)].copy()
# df_it_w = df_it_w[df_it_w['country'].isin(western_countries)].copy()
# df_k_w = df_k_w[df_k_w['country'].isin(western_countries)].copy()
# 
# # Calculate the count of data points for each job category
# group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts1.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts2.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts3.rename(columns={'salary_norm': 'count'}, inplace=True)
# 
# # Specify the minimum number of counts required to keep the group
# min_count = 20
# 
# # Filter groups that meet the criteria
# valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
# valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
# valid_groups3 = group_counts3[group_counts3['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the desired rows
# df_ai_w_c = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_it_w_c = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_k_w_c = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# # List of job categories to exclude
# exclude_categories = ['Architects', 'Leaders', 'Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Team leaders', 
#                       'Out of scope', 'Too vague answers', 'Project managers', 'Other managers']
# 
# # Filter out rows where 'job_category' is in the exclude list
# df_ai_w_c = df_ai_w_c[~df_ai_w_c['job_category'].isin(exclude_categories)]
# df_it_w_c = df_it_w_c[~df_it_w_c['job_category'].isin(exclude_categories)]
# df_k_w_c = df_k_w_c[~df_k_w_c['job_category'].isin(exclude_categories)]


# ## Try: 2024-10-13

# In[121]:


# Filter for senior level
df_ai_w_l_s = df_ai_w_l[df_ai_w_l['seniority_level'] == 'senior'].copy()
df_it_w_l_s = df_it_w_l[df_it_w_l['seniority_level'] == 'senior'].copy()
df_k_w_l_s = df_k_w_l[df_k_w_l['seniority_level'] == 'senior'].copy()

# Function to calculate statistics
def calculate_stats(df, survey_name):
    stats = df.groupby('job_category').agg(
        median=('salary_norm_2024', 'median'),
        p25=('salary_norm_2024', lambda x: x.quantile(0.25)),
        p75=('salary_norm_2024', lambda x: x.quantile(0.75))
    ).reset_index()
    stats['survey'] = survey_name
    return stats

# Calculate stats for each dataset
stats_ai = calculate_stats(df_ai_w_l_s, 'AI')
stats_it = calculate_stats(df_it_w_l_s, 'IT')
stats_k = calculate_stats(df_k_w_l_s, 'K')

# Concatenate all stats
all_stats = pd.concat([stats_ai, stats_it, stats_k], ignore_index=True)

# Compute the maximum median per job_category for ordering
median_max = all_stats.groupby('job_category')['median'].max().reset_index().rename(columns={'median':'median_max'})

# Merge median_max back into all_stats
all_stats = pd.merge(all_stats, median_max, on='job_category', how='left')

# Sort all_stats by median_max descending
all_stats.sort_values(by='median_max', ascending=False, inplace=True)

# Get the ordered list of job categories
job_categories_ordered = all_stats['job_category'].drop_duplicates().tolist()

# Create a mapping from job_category to position
job_category_positions = {job_category: i for i, job_category in enumerate(job_categories_ordered)}

# Bar width and survey offsets
bar_width = 0.2
surveys = ['AI', 'IT', 'K']
offsets = {'AI': -bar_width, 'IT': 0, 'K': bar_width}

# Initialize plot
plt.figure(figsize=(14, 8))

colors = {'AI': 'blue', 'IT': 'green', 'K': 'red'}

for survey in surveys:
    df = all_stats[all_stats['survey'] == survey]
    # Set index to 'job_category' for easy lookup
    df = df.set_index('job_category')
    medians = []
    lower_errors = []
    upper_errors = []
    x_positions = []
    for job_category in job_categories_ordered:
        if job_category in df.index:
            median = df.loc[job_category, 'median']
            p25 = df.loc[job_category, 'p25']
            p75 = df.loc[job_category, 'p75']
            lower_error = median - p25
            upper_error = p75 - median
            position = job_category_positions[job_category] + offsets[survey]
            medians.append(median)
            lower_errors.append(lower_error)
            upper_errors.append(upper_error)
            x_positions.append(position)
        else:
            # No data for this job_category in this survey
            pass
    if medians:
        plt.bar(
            x_positions,
            medians,
            width=bar_width,
            yerr=[lower_errors, upper_errors],
            align='center',
            alpha=0.7,
            ecolor='black',
            capsize=2,
            color=colors[survey],
            label=survey
        )

# Set x-axis labels
x_pos = np.arange(len(job_categories_ordered))
plt.xticks(x_pos, job_categories_ordered, rotation=90)

# Add labels and title
plt.xlabel('Job Category')
plt.ylabel('Normalized Median Salary')
plt.title('Normalized Median Salary per Job Category with 25th and 75th Percentiles')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.show()


# ## Managers vs. Developers

# In[123]:


# # Filters for: western countries, and specific categories
# df_ai_w= df_ai.copy()
# df_it_w= df_it.copy()
# df_k_w = df_k.copy()
# 
# # Western countries
# df_ai_w = df_ai_w[df_ai_w['country'].isin(western_countries)].copy()
# df_it_w = df_it_w[df_it_w['country'].isin(western_countries)].copy()
# df_k_w = df_k_w[df_k_w['country'].isin(western_countries)].copy()
# 
# # Calculate the count of data points for each job category
# group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm_2024'].count().reset_index()
# group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm_2024'].count().reset_index()
# group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm_2024'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts1.rename(columns={'salary_norm_2024': 'count'}, inplace=True)
# group_counts2.rename(columns={'salary_norm_2024': 'count'}, inplace=True)
# group_counts3.rename(columns={'salary_norm_2024': 'count'}, inplace=True)
# 
# # Specify the minimum number of counts required to keep the group
# min_count = 5
# 
# # Filter groups that meet the criteria
# valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
# valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
# valid_groups3 = group_counts3[group_counts3['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the desired rows
# df_ai_w_c = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_it_w_c = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_k_w_c = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# # List of job categories to exclude
# exclude_categories = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers']
# 
# # Filter out rows where 'job_category' is in the exclude list
# df_ai_w_c = df_ai_w_c[~df_ai_w_c['job_category'].isin(exclude_categories)]
# df_it_w_c = df_it_w_c[~df_it_w_c['job_category'].isin(exclude_categories)]
# df_k_w_c = df_k_w_c[~df_k_w_c['job_category'].isin(exclude_categories)]


# In[124]:


# List of managerial categories
managerial_categories = ['Leaders', 'Team leaders', 'Project managers', 'Other managers']

# Create the new column 'developer_job_category'
df_ai_w_l['developer_job_category'] = df_ai_w_l['job_category'].apply(lambda x: 'managerial' if x in managerial_categories else x)
df_it_w_l['developer_job_category'] = df_it_w_l['job_category'].apply(lambda x: 'managerial' if x in managerial_categories else x)
df_k_w_l['developer_job_category'] = df_k_w_l['job_category'].apply(lambda x: 'managerial' if x in managerial_categories else x)

# Create the new column 'managerial_job_category'
df_ai_w_l['managerial_job_category'] = df_ai_w_l['job_category'].apply(lambda x: x if x in managerial_categories else 'developer')
df_it_w_l['managerial_job_category'] = df_it_w_l['job_category'].apply(lambda x: x if x in managerial_categories else 'developer')
df_k_w_l['managerial_job_category'] = df_k_w_l['job_category'].apply(lambda x: x if x in managerial_categories else 'developer')


# ## Try: 2024-10-13

# In[126]:


# Apply the managerial categories as per your code
managerial_categories = ['Leaders', 'Team leaders', 'Project managers', 'Other managers']

# Create 'managerial_job_category' in each DataFrame
for df in [df_ai_w_l, df_it_w_l, df_k_w_l]:
    df['managerial_job_category'] = df['job_category'].apply(
        lambda x: x if x in managerial_categories else 'Developers'
    )

# Filter for 'senior' level
df_ai_w_l_s = df_ai_w_l[df_ai_w_l['seniority_level'] == 'senior'].copy()
df_it_w_l_s = df_it_w_l[df_it_w_l['seniority_level'] == 'senior'].copy()
df_k_w_l_s = df_k_w_l[df_k_w_l['seniority_level'] == 'senior'].copy()

# Function to calculate statistics
def calculate_stats(df, survey_name):
    stats = df.groupby('managerial_job_category').agg(
        median=('salary_norm_2024', 'median'),
        p25=('salary_norm_2024', lambda x: x.quantile(0.25)),
        p75=('salary_norm_2024', lambda x: x.quantile(0.75))
    ).reset_index()
    stats['survey'] = survey_name
    return stats

# Calculate stats for each dataset
stats_ai = calculate_stats(df_ai_w_l_s, 'AI')
stats_it = calculate_stats(df_it_w_l_s, 'IT')
stats_k = calculate_stats(df_k_w_l_s, 'K')

# Concatenate all stats
all_stats = pd.concat([stats_ai, stats_it, stats_k], ignore_index=True)

# Define categories in a specific order
categories = ['Developers', 'Project managers', 'Team leaders', 'Leaders', 'Other managers']

# Create a mapping from category to position
category_positions = {category: idx for idx, category in enumerate(categories)}

# Set up the bar plot
bar_width = 0.2
surveys = ['AI', 'IT', 'K']
colors = {'AI': 'blue', 'IT': 'green', 'K': 'red'}
offsets = {'AI': -bar_width, 'IT': 0, 'K': bar_width}

plt.figure(figsize=(6, 4))

# Initialize a dictionary to keep track of labels added for the legend
label_added = {'AI': False, 'IT': False, 'K': False}

for survey in surveys:
    survey_stats = all_stats[all_stats['survey'] == survey]
    for category in categories:
        cat_stats = survey_stats[survey_stats['managerial_job_category'] == category]
        if not cat_stats.empty:
            median = cat_stats['median'].values[0]
            p25 = cat_stats['p25'].values[0]
            p75 = cat_stats['p75'].values[0]
            lower_error = median - p25
            upper_error = p75 - median
            idx = category_positions[category]
            position = idx + offsets[survey]
            # Determine alpha
            alpha = 0.3 if category != 'Developers' else 1.0
            # Determine label for the legend
            if not label_added[survey]:
                label = survey
                label_added[survey] = True
            else:
                label = ""
            # Plot the bar
            plt.bar(
                position,
                median,
                width=bar_width,
                yerr=[[lower_error], [upper_error]],
                align='center',
                alpha=alpha,
                ecolor='black',
                capsize=5,
                color=colors[survey],
                label=label
            )
        else:
            # No data for this category in this survey
            pass

# Adjust x-axis labels
x_positions = np.arange(len(categories))
plt.xticks(x_positions, categories, rotation=45)
plt.xlabel('Job Category')
plt.ylabel('Normalized Median Salary')
plt.title('Managers vs. Developers Salary Comparison (Senior Level)')

# Add legend
plt.legend()

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)
plt.figure(figsize=(10, 5))
# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()


# ## Germany-IT specific

# ### Experience

# In[129]:


from scipy.stats import pearsonr
from scipy.stats import linregress


# In[130]:


# Prepare the data
df_it_xp = df_it.copy()
df_it_xp = df_it_xp.dropna(subset=['experience', 'salary_2024'])  # Drop rows where either 'experience' or 'salary' is NaN

# Convert experience to numeric and filter values less than 21
df_it_xp['experience'] = df_it_xp['experience'].apply(pd.to_numeric, errors='coerce')
df_it_xp = df_it_xp[df_it_xp['experience'] < 21]

# Drop rows with NaN or Inf values in 'experience' or 'salary'
df_it_xp = df_it_xp[np.isfinite(df_it_xp['experience']) & np.isfinite(df_it_xp['salary_2024'])]

# Perform linear regression to find the slope and intercept
slope, intercept, r_value, p_value, std_err = linregress(df_it_xp['experience'], df_it_xp['salary_2024'])

# Create the scatter plot with a regression line
plt.figure(figsize=(8, 4))
sns.regplot(data=df_it_xp, x='experience', y='salary_2024', 
            scatter_kws={'alpha':0.1, 'edgecolor':'w'}, 
            line_kws={'color':'red', 'lw':2})

# Customize the plot
plt.title('Scatter Plot of Salary vs. Experience with Regression Line')
plt.xlabel('Experience (years)')
plt.ylabel('Salary ($)')
plt.grid(True)
plt.ylim(0, 300000)

# Annotate the plot with the slope and correlation coefficient
plt.text(0.05, 0.95, f'Slope: {slope:.2f}\nCorrelation coefficient: {r_value:.2f}', 
         ha='left', va='top', transform=plt.gca().transAxes, fontsize=12, color='black')

# Show the plot
plt.show()


# ### Years in Germany

# In[132]:


# Prepare the data
df_it_xp = df_it.copy()
df_it_xp = df_it_xp.dropna(subset=['years_of_experience_in_germany', 'salary_2024'])  # Drop rows where either 'experience' or 'salary' is NaN

# Convert experience to numeric and filter values less than 21
df_it_xp['years_of_experience_in_germany'] = df_it_xp['years_of_experience_in_germany'].apply(pd.to_numeric, errors='coerce')
df_it_xp = df_it_xp[df_it_xp['years_of_experience_in_germany'] < 21]

# Drop rows with NaN or Inf values in 'experience' or 'salary'
df_it_xp = df_it_xp[np.isfinite(df_it_xp['years_of_experience_in_germany']) & np.isfinite(df_it_xp['salary_2024'])]

# Perform linear regression to find the slope and intercept
slope, intercept, r_value, p_value, std_err = linregress(df_it_xp['years_of_experience_in_germany'], df_it_xp['salary_2024'])

# Create the scatter plot with a regression line
plt.figure(figsize=(8, 4))
sns.regplot(data=df_it_xp, x='years_of_experience_in_germany', y='salary_2024', 
            scatter_kws={'alpha':0.1, 'edgecolor':'w'}, 
            line_kws={'color':'red', 'lw':2})

# Customize the plot
plt.title('Scatter Plot of Salary vs. Experience with Regression Line')
plt.xlabel('years_of_experience_in_germany (years)')
plt.ylabel('Salary ($)')
plt.grid(True)
plt.ylim(0, 300000)

# Annotate the plot with the slope and correlation coefficient
plt.text(0.05, 0.95, f'Slope: {slope:.2f}\nCorrelation coefficient: {r_value:.2f}', 
         ha='left', va='top', transform=plt.gca().transAxes, fontsize=12, color='black')

# Show the plot
plt.show()


# ### City

# In[134]:


df_it_city = df_it.copy()
df_it_city = df_it_city[(df_it_city['city'] == 'berlin') | (df_it_city['city'] == 'munich')]
avg_values = df_it_city.groupby(['city', 'seniority_level'], observed=True)['salary_2024'].median().reset_index()

plt.figure(figsize=(5, 3))

# Define the order of the seniority levels
seniority_order = ['junior', 'medior', 'senior', 'executive']

# Creating the bar plot
sns.barplot(x='seniority_level', y='salary_2024', hue='city', data=avg_values, order=seniority_order)

# Adding title and labels
plt.title('Average Value per Category')
plt.xlabel('Category')
plt.ylabel('Average Value')

# Display the plot
plt.show()


# ### Language at work

# In[136]:


df_it_l = df_it.copy()
avg_values = df_it_l.groupby(['language_category', 'seniority_level'], observed=True)['salary_2024'].median().reset_index()

plt.figure(figsize=(5, 3))

# Define the order of the seniority levels
seniority_order = ['junior', 'medior', 'senior', 'executive']

# Creating the bar plot
sns.barplot(x='seniority_level', y='salary_2024', hue='language_category', data=avg_values, order=seniority_order)

# Adding title and labels
plt.title('Average Value per Category')
plt.xlabel('Category')
plt.ylabel('Average Value')

# Display the plot
plt.show()


# ## Kaggle-Specific

# ### BSc/MSc/PhD

# In[139]:


df_k.education_level.unique()


# In[140]:


df_k_e = df_k_w.copy()
# df_k_e = df_k_e[(df_k_e['country'] == 'us')]
# df_k_e['education_level'] = df_k_e['education_level'].astype('string')
# df_k_e['education_level'] = df_k_e['education_level'].str.replace('no formal education past high school','no degree')
df_k_e = df_k_e[(df_k_e['education_level'] == 'bachelor’s degree') | (df_k_e['education_level'] == 'master’s degree') | (df_k_e['education_level'] == 'doctoral degree') | (df_k_e['education_level'] == 'no degree')]
median_values = df_k_e.groupby(['education_level', 'seniority_level'], observed=True)['salary_norm_2024'].median().reset_index()

plt.figure(figsize=(8, 3))

# Define the order for plotting
seniority_order = ['junior', 'medior', 'senior', 'executive']
education_order = ['no degree', 'bachelor’s degree', 'master’s degree', 'doctoral degree']

# Creating the bar plot
sns.barplot(x='seniority_level', y='salary_norm_2024', hue='education_level', data=median_values, order=seniority_order, hue_order=education_order)

# Adding title and labels
plt.title('Average Value per Category')
plt.xlabel('Category')
plt.ylabel('Average Value')

# Display the plot
plt.show()


# # Factorial cell visualization

# ## AI-Jobs.net

# In[143]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_ai.pivot_table(index=['country', 'seniority_level'], columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(12, 16))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[144]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_ai.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(10, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[145]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_ai_w_l.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(10, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[146]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_ai_data.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(10, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# ## De-IT

# In[148]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_it.pivot_table(index=['country', 'seniority_level'], columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(14, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[149]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_it_w_l.pivot_table(index=['country', 'seniority_level'], columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(14, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[150]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_it_data.pivot_table(index=['country', 'seniority_level'], columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(14, 3))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# ## Kaggle

# In[152]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_k.pivot_table(index=['country', 'seniority_level'], columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(8, 30))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[153]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_k.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[154]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_k_w_l.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# In[155]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_k_data.pivot_table(index='seniority_level', columns='job_category', aggfunc='size', fill_value=0, observed=True)

# Plot the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# Conclusion: \
# Only data-related job-titles are sufficiently represented in the factorial cells. \
# And only if we eliminate the countries by normalizing with GPD-per-capita. \
# Executive positions are out of scope, those are disjointed from the job-titles.

# In[157]:


# Create a pivot table to count the number of observations in each factorial cell
pivot_table = df_data.pivot_table(index=['survey','seniority_level'], columns='job_category', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='viridis')
plt.title('Number of Datapoints in Each Factorial Cell for Survey K')
plt.xlabel('JobGroup')
plt.ylabel('Country and Seniority')
plt.show()


# # Preparing dataframes

# In[159]:


# # First we filter for western countries
# df_ai_w= df_ai.copy()
# df_it_w= df_it.copy()
# df_k_w = df_k.copy()
# 
# # Western countries
# df_ai_w = df_ai_w[df_ai_w['country'].isin(western_countries)].copy()
# df_it_w = df_it_w[df_it_w['country'].isin(western_countries)].copy()
# df_k_w = df_k_w[df_k_w['country'].isin(western_countries)].copy()


# In[160]:


# # List of job categories to exclude
# exclude_categories = ['Consultant', '"Other"', 'Uncategorized', 'Advocacy', 'Out of scope', 'Too vague answers', 'Other managers']
# #Not excluding now: ['Architects', 'Leaders', 'Project managers', 'Team leaders']
# 
# # Filter out rows where 'job_category' is in the exclude list
# df_ai_w = df_ai_w[~df_ai_w['job_category'].isin(exclude_categories)]
# df_it_w = df_it_w[~df_it_w['job_category'].isin(exclude_categories)]
# df_k_w = df_k_w[~df_k_w['job_category'].isin(exclude_categories)]


# ## Filtering for Factorial Cell's length

# Filtering out rows, which would result in Factorial Groups with less datapoint than the statistically required threshold

# ### For Kolmogorov-Smirnov

# In[164]:


# # Specify the minimum number of counts required to keep the group
# min_count = 30
# 
# # Calculate the count of data points for each job category
# group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts1.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts2.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts3.rename(columns={'salary_norm': 'count'}, inplace=True)
# 
# # Filter groups that meet the criteria
# valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
# valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
# valid_groups3 = group_counts3[group_counts3['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the desired rows
# df_ai_w_ks = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_it_w_ks = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_k_w_ks = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')


# #### Listing out the eligible factorial groups for K-S

# In[166]:


# # Extract unique (job_category, seniority_level) pairs from each dataframe
# groups_ai_factorial = set(df_ai_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
# groups_k_factorial = set(df_k_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
# groups_it_factorial = set(df_it_w_l[['job_category', 'seniority_level']].drop_duplicates().itertuples(index=False, name=None))
# 
# groups_ai_factorial_list = sorted(list(groups_ai_factorial))
# groups_k_factorial_list = sorted(list(groups_k_factorial))
# groups_it_factorial_list = sorted(list(groups_it_factorial))


# ### For Levene (or ANOVA)

# In[168]:


# # Specify the minimum number of counts required to keep the group
# min_count = 20
# 
# # Calculate the count of data points for each job category
# group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts1.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts2.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts3.rename(columns={'salary_norm': 'count'}, inplace=True)
# 
# # Filter groups that meet the criteria
# valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
# valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
# valid_groups3 = group_counts3[group_counts3['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the desired rows
# df_ai_w_l = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_it_w_l = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_k_w_l = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')


# In[169]:


# # Specify the minimum number of counts required to keep the group
# min_count = 20
# 
# # Calculate the count of data points for each combination of seniority_level and job_category
# group_counts = df_combined.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts.rename(columns={'salary_norm': 'count'}, inplace=True)
# 
# # Filter groups that meet the minimum count criteria
# valid_groups = group_counts[group_counts['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the rows from the valid combinations
# df_combined_filtered = pd.merge(df_combined, valid_groups[['seniority_level', 'job_category']], 
#                                 on=['seniority_level', 'job_category'], 
#                                 how='inner')


# ### Spoiler: For Kruskal-Wallis

# In[171]:


# # Specify the minimum number of counts required to keep the group
# min_count = 10
# 
# # Calculate the count of data points for each job category
# group_counts1 = df_ai_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts2 = df_it_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# group_counts3 = df_k_w.groupby(['seniority_level', 'job_category'], observed=True)['salary_norm'].count().reset_index()
# 
# # Rename the count column for clarity
# group_counts1.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts2.rename(columns={'salary_norm': 'count'}, inplace=True)
# group_counts3.rename(columns={'salary_norm': 'count'}, inplace=True)
# 
# # Filter groups that meet the criteria
# valid_groups1 = group_counts1[group_counts1['count'] >= min_count]
# valid_groups2 = group_counts2[group_counts2['count'] >= min_count]
# valid_groups3 = group_counts3[group_counts3['count'] >= min_count]
# 
# # Merge the valid groups back with the original dataframe to keep only the desired rows
# df_ai_w_kw = pd.merge(df_ai_w, valid_groups1[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_it_w_kw = pd.merge(df_it_w, valid_groups2[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')
# 
# df_k_w_kw = pd.merge(df_k_w, valid_groups3[['seniority_level', 'job_category']], 
#                        on=['seniority_level', 'job_category'], 
#                        how='inner')


# # Distribution function

# In[173]:


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


# In[174]:


def assess_fit(data, min_sample_size=5):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=5):
    fit_counts = 0
    total_groups = 0
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for _, group in grouped:
        total_groups += 1
        if assess_fit(group['salary_norm'], min_sample_size):
            fit_counts += 1
    return fit_counts, total_groups

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_counts, total_groups = evaluate_survey(df)
    results[survey_name] = {
        'fit_counts': fit_counts,
        'total_groups': total_groups,
        'fit_percentage': (fit_counts / total_groups) * 100 if total_groups > 0 else 0
    }

# Print results
for survey_name, result in results.items():
    print(f"{survey_name}: {result['fit_counts']} out of {result['total_groups']} groups fit the log-normal distribution ({result['fit_percentage']:.2f}%)")


# In[175]:


def assess_fit(data, min_sample_size=5):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=5):
    fit_counts = 0
    total_groups = 0
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for _, group in grouped:
        total_groups += 1
        if assess_fit(group['salary'], min_sample_size):
            fit_counts += 1
    return fit_counts, total_groups

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_counts, total_groups = evaluate_survey(df)
    results[survey_name] = {
        'fit_counts': fit_counts,
        'total_groups': total_groups,
        'fit_percentage': (fit_counts / total_groups) * 100 if total_groups > 0 else 0
    }

# Print results
for survey_name, result in results.items():
    print(f"{survey_name}: {result['fit_counts']} out of {result['total_groups']} groups fit the normal distribution ({result['fit_percentage']:.2f}%)")


# In[176]:


def assess_fit(data, min_sample_size=10):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=10):
    fit_results = {
        'fits': [],
        'non_fits': []
    }
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        if assess_fit(group['salary_norm'], min_sample_size):
            fit_results['fits'].append(name)
        else:
            fit_results['non_fits'].append(name)
    return fit_results

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_results = evaluate_survey(df)
    results[survey_name] = fit_results

# Print detailed results
for survey_name, fit_results in results.items():
    fits = fit_results['fits']
    non_fits = fit_results['non_fits']
    total_groups = len(fits) + len(non_fits)
    fit_percentage = (len(fits) / total_groups) * 100 if total_groups > 0 else 0
    
    print(f"{survey_name}:")
    #print(f"Total groups: {total_groups}")
    print(f"Fit groups ({len(fits)}): {fits}")
    #print(f"Non-fit groups ({len(non_fits)}): {non_fits}")
    print(f"Fit percentage: {fit_percentage:.2f}%\n")


# In[177]:


def assess_fit(data, min_sample_size=5):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=5):
    fit_results = {
        'fits': [],
        'non_fits': []
    }
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        if assess_fit(group['salary_norm'], min_sample_size):
            fit_results['fits'].append(name)
        else:
            fit_results['non_fits'].append(name)
    return fit_results

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_results = evaluate_survey(df)
    results[survey_name] = fit_results

# Print detailed results
for survey_name, fit_results in results.items():
    fits = fit_results['fits']
    non_fits = fit_results['non_fits']
    total_groups = len(fits) + len(non_fits)
    fit_percentage = (len(fits) / total_groups) * 100 if total_groups > 0 else 0
    
    print(f"{survey_name}:")
    #print(f"Total groups: {total_groups}")
    print(f"Fit groups ({len(fits)}): {fits}")
    #print(f"Non-fit groups ({len(non_fits)}): {non_fits}")
    print(f"Fit percentage: {fit_percentage:.2f}%\n")


# In[178]:


def assess_fit_normal(data, min_sample_size):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_lognormal(data, min_sample_size):
    if len(data) < min_sample_size:
        return False  # Insufficient data
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=10):
    fit_results = {
        'fits': [],
        'non_fits': []
    }
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['seniority_level','year'], observed=True)
    for name, group in grouped:
        if assess_fit_normal(group['salary_norm'], min_sample_size):
            fit_results['fits'].append(name)
        else:
            fit_results['non_fits'].append(name)
    return fit_results

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_results = evaluate_survey(df)
    results[survey_name] = fit_results

# Print detailed results
for survey_name, fit_results in results.items():
    fits = fit_results['fits']
    non_fits = fit_results['non_fits']
    total_groups = len(fits) + len(non_fits)
    fit_percentage = (len(fits) / total_groups) * 100 if total_groups > 0 else 0
    
    print(f"{survey_name}:")
    print(f"Total groups: {total_groups}")
    print(f"Fit groups ({len(fits)}): {fits}")
    #print(f"Non-fit groups ({len(non_fits)}): {non_fits}")
    print(f"Fit percentage: {fit_percentage:.2f}%\n")


# In[179]:


def assess_fit_normal(data, min_sample_size):
    if len(data) < min_sample_size:
        return None  # Insufficient data
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_lognormal(data, min_sample_size):
    if len(data) < min_sample_size:
        return None  # Insufficient data
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=100, fit_function=assess_fit_normal):
    fit_results = {
        'fits': [],
        'non_fits': [],
        'eligible_groups': 0
    }
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        fit = fit_function(group['salary_norm'], min_sample_size)
        if fit is not None:  # Only consider groups with sufficient data
            fit_results['eligible_groups'] += 1
            if fit:
                fit_results['fits'].append(name)
            else:
                fit_results['non_fits'].append(name)
    return fit_results

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_results = evaluate_survey(df)
    results[survey_name] = fit_results

# Print detailed results
for survey_name, fit_results in results.items():
    fits = fit_results['fits']
    non_fits = fit_results['non_fits']
    eligible_groups = fit_results['eligible_groups']
    fit_percentage = (len(fits) / eligible_groups) * 100 if eligible_groups > 0 else 0
    
    print(f"{survey_name}:")
    print(f"Eligible groups: {eligible_groups}")
    print(f"Fit groups ({len(fits)}): {fits}")
    #print(f"Non-fit groups ({len(non_fits)}): {non_fits}")
    print(f"Fit percentage: {fit_percentage:.2f}%\n")


# In[180]:


def assess_fit_normal(data, min_sample_size):
    if len(data) < min_sample_size:
        return None  # Insufficient data
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_lognormal(data, min_sample_size):
    if len(data) < min_sample_size:
        return None  # Insufficient data
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def evaluate_survey(df, min_sample_size=30, fit_function=assess_fit_normal):
    fit_results = {
        'fits': [],
        'non_fits': [],
        'eligible_groups': 0
    }
    # Group by year, job_category, and seniority_level
    grouped = df.groupby(['country', 'job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        sample_size = len(group['salary'])
        fit = fit_function(group['salary'], min_sample_size)
        if fit is not None:  # Only consider groups with sufficient data
            fit_results['eligible_groups'] += 1
            if fit:
                fit_results['fits'].append((name, sample_size))
            else:
                fit_results['non_fits'].append((name, sample_size))
    return fit_results

# Load your surveys (assume they are already loaded as df_it, df_k, df_ai)
# df_it = pd.read_csv('path_to_it_survey.csv')
# df_k = pd.read_csv('path_to_k_survey.csv')
# df_ai = pd.read_csv('path_to_ai_survey.csv')

surveys = {'df_it': df_it, 'df_k': df_k, 'df_ai': df_ai}
results = {}

for survey_name, df in surveys.items():
    fit_results = evaluate_survey(df)
    results[survey_name] = fit_results

# Print detailed results
for survey_name, fit_results in results.items():
    fits = fit_results['fits']
    non_fits = fit_results['non_fits']
    eligible_groups = fit_results['eligible_groups']
    fit_percentage = (len(fits) / eligible_groups) * 100 if eligible_groups > 0 else 0
    
    print(f"{survey_name}:")
    print(f"Eligible groups: {eligible_groups}")
    print(f"Fit groups ({len(fits)}):")
    for name, sample_size in fits:
        print(f"  {name} (sample size: {sample_size})")
    #print(f"Non-fit groups ({len(non_fits)}):")
    #for name, sample_size in non_fits:
    #    print(f"  {name} (sample size: {sample_size})")
    print(f"Fit percentage: {fit_percentage:.2f}%\n")


# ## Q-Q plot

# In[182]:


def plot_qq_plots(df, job_category, seniority_level, country, year, dot_size):
    # Filter the dataframe for the specific group
    group = df[(df['job_category'] == job_category) & 
               (df['seniority_level'] == seniority_level) #& 
               #(df['country'] == country) & 
               #(df['year'] == year)
              ]
    
    if group.empty:
        print("No data available for the specified group.")
        return
    
    data = group['salary_norm_2024']
    
    # Fit distributions to the data
    mean, std = data.mean(), data.std()
    shape_log, loc_log, scale_log = stats.lognorm.fit(data, floc=0)
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data, floc=0)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Q-Q Plots for Various Distributions\n{job_category} ({seniority_level}) in {country}, {year}', fontsize=16)
    
    # Normal Q-Q plot
    theoretical_quantiles = np.linspace(0, 1, len(data))
    normal_quantiles = stats.norm.ppf(theoretical_quantiles, mean, std)
    axes[0, 0].scatter(normal_quantiles, np.sort(data), s=dot_size, color='blue', edgecolor='black')
    axes[0, 0].plot(normal_quantiles, normal_quantiles, 'r-', lw=2)  # Reference line
    axes[0, 0].set_title('Normal Distribution')
    axes[0, 0].set_xlabel('Theoretical Quantiles')
    axes[0, 0].set_ylabel('Sample Quantiles')
    
    # Lognormal Q-Q plot
    sorted_data = np.sort(data)
    norm_quantiles = stats.norm.ppf((np.arange(len(data)) + 0.5) / len(data))
    #lognormal_quantiles = np.exp(norm_quantiles * std + mean)  # Using original scale of quantiles
    lognormal_quantiles = np.exp(norm_quantiles * shape_log + np.log(scale_log))
    axes[0, 1].scatter(lognormal_quantiles, sorted_data, s=dot_size, color='blue', edgecolor='black')
    axes[0, 1].plot(lognormal_quantiles, lognormal_quantiles, 'r-', lw=2)  # Reference line
    axes[0, 1].set_title('Lognormal Distribution')
    axes[0, 1].set_xlabel('Theoretical Quantiles')
    axes[0, 1].set_ylabel('Sample Quantiles')
    
    # Gamma Q-Q plot
    sorted_data = np.sort(data)
    theoretical_quantiles_gamma = stats.gamma.ppf((np.arange(len(data)) + 0.5) / len(data), shape_gamma, loc_gamma, scale_gamma)
    axes[1, 0].scatter(theoretical_quantiles_gamma, sorted_data, s=dot_size, color='blue', edgecolor='black')
    axes[1, 0].plot(theoretical_quantiles_gamma, theoretical_quantiles_gamma, 'r-', lw=2)  # Reference line
    axes[1, 0].set_title('Gamma Distribution')
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Sample Quantiles')
    
    # Weibull Q-Q plot
    sorted_data = np.sort(data)
    theoretical_quantiles_weibull = stats.weibull_min.ppf((np.arange(len(data)) + 0.5) / len(data), shape_weibull, loc_weibull, scale_weibull)
    axes[1, 1].scatter(theoretical_quantiles_weibull, sorted_data, s=dot_size, color='blue', edgecolor='black')
    axes[1, 1].plot(theoretical_quantiles_weibull, theoretical_quantiles_weibull, 'r-', lw=2)  # Reference line
    axes[1, 1].set_title('Weibull Distribution')
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Show the plot
    plt.show()

# Example usage with one of the surveys (assuming df_ai is loaded)
# Adjust the parameters as needed
plot_qq_plots(df_ai, job_category='Data Analyst', seniority_level='senior', country='us', year='2020', dot_size=5)


# ## Fitting distribution-functions to specific factorial cells

# In[184]:


def calculate_r_squared(data, pdf, params):
    observed = data
    expected = pdf(data, *params)
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)

def goodness_of_fit(data, dist_name, params):
    ks_statistic, ks_pvalue = stats.kstest(data, dist_name, args=params)
    return ks_statistic, ks_pvalue

def plot_distribution(df, job_category, seniority_level, country, year):
    # Filter the dataframe for the specific group
    group = df[
               (df['job_category'] == job_category) & 
               (df['seniority_level'] == seniority_level) #&
               #(df['country'] == country) #&
               #(df['year'] == year) &
              ]
    
    if group.empty:
        print("No data available for the specified group.")
        return
    
    data = group['salary_norm_2024']
    
    # Calculate the parameters for the normal, lognormal, Pareto, gamma, and Weibull distributions
    mean, std = data.mean(), data.std()
    shape_log, loc_log, scale_log = stats.lognorm.fit(data, floc=0)
    b, loc_pareto, scale_pareto = stats.pareto.fit(data, floc=0)
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data, floc=0)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    
    # Calculate goodness-of-fit metrics
    normal_r2 = calculate_r_squared(data, stats.norm.pdf, (mean, std))
    lognormal_r2 = calculate_r_squared(data, stats.lognorm.pdf, (shape_log, loc_log, scale_log))
    pareto_r2 = calculate_r_squared(data, stats.pareto.pdf, (b, loc_pareto, scale_pareto))
    gamma_r2 = calculate_r_squared(data, stats.gamma.pdf, (shape_gamma, loc_gamma, scale_gamma))
    weibull_r2 = calculate_r_squared(data, stats.weibull_min.pdf, (shape_weibull, loc_weibull, scale_weibull))
    
    normal_ks = goodness_of_fit(data, 'norm', (mean, std))
    lognormal_ks = goodness_of_fit(data, 'lognorm', (shape_log, loc_log, scale_log))
    pareto_ks = goodness_of_fit(data, 'pareto', (b, loc_pareto, scale_pareto))
    gamma_ks = goodness_of_fit(data, 'gamma', (shape_gamma, loc_gamma, scale_gamma))
    weibull_ks = goodness_of_fit(data, 'weibull_min', (shape_weibull, loc_weibull, scale_weibull))
    
    # Generate points on the x axis for plotting the fitted curves
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Plot the histogram
    plt.figure(figsize=(10, 3))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    
    # Plot the fitted normal distribution curve
    plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2, label=f'Normal, R^2={normal_r2:.4f}, KS={normal_ks[0]:.4f}, p={normal_ks[1]:.4f}')
    
    # Plot the fitted lognormal distribution curve
    plt.plot(x, stats.lognorm.pdf(x, shape_log, loc_log, scale_log), 'b-', lw=2, label=f'Lognormal, R^2={lognormal_r2:.4f}, KS={lognormal_ks[0]:.4f}, p={lognormal_ks[1]:.4f}')
    
    # Plot the fitted Pareto distribution curve
    plt.plot(x, stats.pareto.pdf(x, b, loc_pareto, scale_pareto), 'y-', lw=2, label=f'Pareto, R^2={pareto_r2:.4f}, KS={pareto_ks[0]:.4f}, p={pareto_ks[1]:.4f}')
    
    # Plot the fitted gamma distribution curve
    plt.plot(x, stats.gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'm-', lw=2, label=f'Gamma, R^2={gamma_r2:.4f}, KS={gamma_ks[0]:.4f}, p={gamma_ks[1]:.4f}')
    
    # Plot the fitted Weibull distribution curve
    plt.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'c-', lw=2, label=f'Weibull, R^2={weibull_r2:.4f}, KS={weibull_ks[0]:.4f}, p={weibull_ks[1]:.4f}')
    
    # Add titles and labels
    plt.title(f'Distribution of Salaries for {job_category} ({seniority_level})')
    plt.xlabel('Salary (normalized with GDP-per-capita)')
    plt.ylabel('Density')
    plt.legend()
    
    # Adjust the y-limit to focus on the histogram, normal, and lognormal curves
    hist_max = np.histogram(data, bins=30, density=True)[0].max()
    plt.ylim(0, max(hist_max, max(stats.norm.pdf(x, mean, std)), max(stats.lognorm.pdf(x, shape_log, loc_log, scale_log))) * 1.2)
    
    # Show the plot
    plt.show()

# Example usage with one of the surveys (assuming df_it, df_k, df_ai are loaded)
# Adjust the parameters as needed
plot_distribution(df_it, job_category='Data Analyst', seniority_level='senior', country='', year='')


# In[185]:


def calculate_r_squared(data, pdf, params):
    observed = data
    expected = pdf(data, *params)
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)

def goodness_of_fit(data, dist_name, params):
    ks_statistic, ks_pvalue = stats.kstest(data, dist_name, args=params)
    return ks_statistic, ks_pvalue

def plot_distribution(df, job_category, seniority_level, country, year):
    # Filter the dataframe for the specific group
    group = df[
               (df['job_category'] == job_category) & 
               (df['seniority_level'] == seniority_level) #&
               #(df['country'] == country) #&
               #(df['year'] == year) &
              ]
    
    if group.empty:
        print("No data available for the specified group.")
        return
    
    data = group['salary_norm']
    
    # Calculate the parameters for the normal, lognormal, Pareto, gamma, and Weibull distributions
    mean, std = data.mean(), data.std()
    shape_log, loc_log, scale_log = stats.lognorm.fit(data, floc=0)
    b, loc_pareto, scale_pareto = stats.pareto.fit(data, floc=0)
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data, floc=0)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    
    # Calculate goodness-of-fit metrics
    normal_r2 = calculate_r_squared(data, stats.norm.pdf, (mean, std))
    lognormal_r2 = calculate_r_squared(data, stats.lognorm.pdf, (shape_log, loc_log, scale_log))
    pareto_r2 = calculate_r_squared(data, stats.pareto.pdf, (b, loc_pareto, scale_pareto))
    gamma_r2 = calculate_r_squared(data, stats.gamma.pdf, (shape_gamma, loc_gamma, scale_gamma))
    weibull_r2 = calculate_r_squared(data, stats.weibull_min.pdf, (shape_weibull, loc_weibull, scale_weibull))
    
    normal_ks = goodness_of_fit(data, 'norm', (mean, std))
    lognormal_ks = goodness_of_fit(data, 'lognorm', (shape_log, loc_log, scale_log))
    pareto_ks = goodness_of_fit(data, 'pareto', (b, loc_pareto, scale_pareto))
    gamma_ks = goodness_of_fit(data, 'gamma', (shape_gamma, loc_gamma, scale_gamma))
    weibull_ks = goodness_of_fit(data, 'weibull_min', (shape_weibull, loc_weibull, scale_weibull))
    
    # Generate points on the x axis for plotting the fitted curves
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Plot the histogram
    plt.figure(figsize=(10, 3))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    
    # Plot the fitted normal distribution curve
    plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2, label=f'Normal, R^2={normal_r2:.4f}, KS={normal_ks[0]:.4f}, p={normal_ks[1]:.4f}')
    
    # Plot the fitted lognormal distribution curve
    plt.plot(x, stats.lognorm.pdf(x, shape_log, loc_log, scale_log), 'b-', lw=2, label=f'Lognormal, R^2={lognormal_r2:.4f}, KS={lognormal_ks[0]:.4f}, p={lognormal_ks[1]:.4f}')
    
    # Plot the fitted Pareto distribution curve
    plt.plot(x, stats.pareto.pdf(x, b, loc_pareto, scale_pareto), 'y-', lw=2, label=f'Pareto, R^2={pareto_r2:.4f}, KS={pareto_ks[0]:.4f}, p={pareto_ks[1]:.4f}')
    
    # Plot the fitted gamma distribution curve
    plt.plot(x, stats.gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'm-', lw=2, label=f'Gamma, R^2={gamma_r2:.4f}, KS={gamma_ks[0]:.4f}, p={gamma_ks[1]:.4f}')
    
    # Plot the fitted Weibull distribution curve
    plt.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'c-', lw=2, label=f'Weibull, R^2={weibull_r2:.4f}, KS={weibull_ks[0]:.4f}, p={weibull_ks[1]:.4f}')
    
    # Add titles and labels
    plt.title(f'Distribution of Salaries for {job_category} ({seniority_level})')
    plt.xlabel('Salary (normalized with GDP-per-capita)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Adjust the y-limit to focus on the histogram, normal, and lognormal curves
    hist_max = np.histogram(data, bins=30, density=True)[0].max()
    plt.ylim(0, max(hist_max, max(stats.norm.pdf(x, mean, std)), max(stats.lognorm.pdf(x, shape_log, loc_log, scale_log))) * 1.2)
    
    # Show the plot
    plt.show()

# Example usage with one of the surveys (assuming df_it, df_k, df_ai are loaded)
# Adjust the parameters as needed
plot_distribution(df_ai_w, job_category='Data Analyst', seniority_level='senior', country='', year='')


# In[186]:


def calculate_r_squared(data, pdf, params):
    observed = data
    expected = pdf(data, *params)
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)

def goodness_of_fit(data, dist_name, params):
    ks_statistic, ks_pvalue = stats.kstest(data, dist_name, args=params)
    return ks_statistic, ks_pvalue

def plot_distribution(df, job_category, seniority_level, country, year):
    # Filter the dataframe for the specific group
    group = df[
               (df['job_category'] == job_category) & 
               (df['seniority_level'] == seniority_level) &
               (df['country'] == country) #&
               #(df['year'] == year) &
              ]
    
    if group.empty:
        print("No data available for the specified group.")
        return
    
    data = group['salary_norm']
    
    # Calculate the parameters for the normal, lognormal, Pareto, gamma, and Weibull distributions
    mean, std = data.mean(), data.std()
    shape_log, loc_log, scale_log = stats.lognorm.fit(data, floc=0)
    b, loc_pareto, scale_pareto = stats.pareto.fit(data, floc=0)
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data, floc=0)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    
    # Calculate goodness-of-fit metrics
    normal_r2 = calculate_r_squared(data, stats.norm.pdf, (mean, std))
    lognormal_r2 = calculate_r_squared(data, stats.lognorm.pdf, (shape_log, loc_log, scale_log))
    pareto_r2 = calculate_r_squared(data, stats.pareto.pdf, (b, loc_pareto, scale_pareto))
    gamma_r2 = calculate_r_squared(data, stats.gamma.pdf, (shape_gamma, loc_gamma, scale_gamma))
    weibull_r2 = calculate_r_squared(data, stats.weibull_min.pdf, (shape_weibull, loc_weibull, scale_weibull))
    
    normal_ks = goodness_of_fit(data, 'norm', (mean, std))
    lognormal_ks = goodness_of_fit(data, 'lognorm', (shape_log, loc_log, scale_log))
    pareto_ks = goodness_of_fit(data, 'pareto', (b, loc_pareto, scale_pareto))
    gamma_ks = goodness_of_fit(data, 'gamma', (shape_gamma, loc_gamma, scale_gamma))
    weibull_ks = goodness_of_fit(data, 'weibull_min', (shape_weibull, loc_weibull, scale_weibull))
    
    # Generate points on the x axis for plotting the fitted curves
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Plot the histogram
    plt.figure(figsize=(10, 3))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    
    # Plot the fitted normal distribution curve
    plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2, label=f'Normal, R^2={normal_r2:.4f}, KS={normal_ks[0]:.4f}, p={normal_ks[1]:.4f}')
    
    # Plot the fitted lognormal distribution curve
    plt.plot(x, stats.lognorm.pdf(x, shape_log, loc_log, scale_log), 'b-', lw=2, label=f'Lognormal, R^2={lognormal_r2:.4f}, KS={lognormal_ks[0]:.4f}, p={lognormal_ks[1]:.4f}')
    
    # Plot the fitted Pareto distribution curve
    plt.plot(x, stats.pareto.pdf(x, b, loc_pareto, scale_pareto), 'y-', lw=2, label=f'Pareto, R^2={pareto_r2:.4f}, KS={pareto_ks[0]:.4f}, p={pareto_ks[1]:.4f}')
    
    # Plot the fitted gamma distribution curve
    plt.plot(x, stats.gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'm-', lw=2, label=f'Gamma, R^2={gamma_r2:.4f}, KS={gamma_ks[0]:.4f}, p={gamma_ks[1]:.4f}')
    
    # Plot the fitted Weibull distribution curve
    plt.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'c-', lw=2, label=f'Weibull, R^2={weibull_r2:.4f}, KS={weibull_ks[0]:.4f}, p={weibull_ks[1]:.4f}')
    
    # Add titles and labels
    plt.title(f'Distribution of Salaries for {job_category} ({seniority_level})')
    plt.xlabel('Salary (normalized with GDP-per-capita)')
    plt.ylabel('Density')
    plt.legend()
    
    # Adjust the y-limit to focus on the histogram, normal, and lognormal curves
    hist_max = np.histogram(data, bins=30, density=True)[0].max()
    plt.ylim(0, max(hist_max, max(stats.norm.pdf(x, mean, std)), max(stats.lognorm.pdf(x, shape_log, loc_log, scale_log))) * 1.2)
    
    # Show the plot
    plt.show()

# Example usage with one of the surveys (assuming df_it, df_k, df_ai are loaded)
# Adjust the parameters as needed
plot_distribution(df_ai, job_category='Data Scientist/ ML Engineer', seniority_level='senior', country='us', year='')


# #### Trying to confirm that lognorm is not the best overall fit?

# In[188]:


def calculate_r_squared(data, pdf, params):
    observed = data
    expected = pdf(data, *params)
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)

def goodness_of_fit(data, dist_name, params):
    ks_statistic, ks_pvalue = stats.kstest(data, dist_name, args=params)
    return ks_statistic, ks_pvalue

def plot_distribution(df, job_category, seniority_level, country, year):
    # Filter the dataframe for the specific group
    group = df[
               (df['job_category'] == job_category) & 
               (df['seniority_level'] == seniority_level) #&
               #(df['country'] == country) #&
               #(df['year'] == year) &
              ]
    
    if group.empty:
        print("No data available for the specified group.")
        return
    
    data = group['salary_norm']
    
    # Calculate the parameters for the normal, lognormal, Pareto, gamma, and Weibull distributions
    mean, std = data.mean(), data.std()
    shape_log, loc_log, scale_log = stats.lognorm.fit(data, floc=0)
    b, loc_pareto, scale_pareto = stats.pareto.fit(data, floc=0)
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data, floc=0)
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(data, floc=0)
    
    # Calculate goodness-of-fit metrics
    normal_r2 = calculate_r_squared(data, stats.norm.pdf, (mean, std))
    lognormal_r2 = calculate_r_squared(data, stats.lognorm.pdf, (shape_log, loc_log, scale_log))
    pareto_r2 = calculate_r_squared(data, stats.pareto.pdf, (b, loc_pareto, scale_pareto))
    gamma_r2 = calculate_r_squared(data, stats.gamma.pdf, (shape_gamma, loc_gamma, scale_gamma))
    weibull_r2 = calculate_r_squared(data, stats.weibull_min.pdf, (shape_weibull, loc_weibull, scale_weibull))
    
    normal_ks = goodness_of_fit(data, 'norm', (mean, std))
    lognormal_ks = goodness_of_fit(data, 'lognorm', (shape_log, loc_log, scale_log))
    pareto_ks = goodness_of_fit(data, 'pareto', (b, loc_pareto, scale_pareto))
    gamma_ks = goodness_of_fit(data, 'gamma', (shape_gamma, loc_gamma, scale_gamma))
    weibull_ks = goodness_of_fit(data, 'weibull_min', (shape_weibull, loc_weibull, scale_weibull))
    
    # Generate points on the x axis for plotting the fitted curves
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Plot the histogram
    plt.figure(figsize=(10, 3))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    
    # Plot the fitted normal distribution curve
    plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2, label=f'Normal, R^2={normal_r2:.4f}, KS={normal_ks[0]:.4f}, p={normal_ks[1]:.4f}')
    
    # Plot the fitted lognormal distribution curve
    plt.plot(x, stats.lognorm.pdf(x, shape_log, loc_log, scale_log), 'b-', lw=2, label=f'Lognormal, R^2={lognormal_r2:.4f}, KS={lognormal_ks[0]:.4f}, p={lognormal_ks[1]:.4f}')
    
    # Plot the fitted Pareto distribution curve
    plt.plot(x, stats.pareto.pdf(x, b, loc_pareto, scale_pareto), 'y-', lw=2, label=f'Pareto, R^2={pareto_r2:.4f}, KS={pareto_ks[0]:.4f}, p={pareto_ks[1]:.4f}')
    
    # Plot the fitted gamma distribution curve
    plt.plot(x, stats.gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'm-', lw=2, label=f'Gamma, R^2={gamma_r2:.4f}, KS={gamma_ks[0]:.4f}, p={gamma_ks[1]:.4f}')
    
    # Plot the fitted Weibull distribution curve
    plt.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'c-', lw=2, label=f'Weibull, R^2={weibull_r2:.4f}, KS={weibull_ks[0]:.4f}, p={weibull_ks[1]:.4f}')
    
    # Add titles and labels
    plt.title(f'Distribution of Salaries for {job_category} ({seniority_level})')
    plt.xlabel('Salary (normalized with GDP-per-capita)')
    plt.ylabel('Density')
    plt.legend()
    
    # Adjust the y-limit to focus on the histogram, normal, and lognormal curves
    hist_max = np.histogram(data, bins=30, density=True)[0].max()
    plt.ylim(0, max(hist_max, max(stats.norm.pdf(x, mean, std)), max(stats.lognorm.pdf(x, shape_log, loc_log, scale_log))) * 1.2)
    
    # Show the plot
    plt.show()

# Example usage with one of the surveys (assuming df_it, df_k, df_ai are loaded)
# Adjust the parameters as needed
plot_distribution(df_k_w_l, job_category='Database Dev & Admin', seniority_level='senior', country='', year='')


# ## Fitting the distribution-functions to all factorial cells

# ##### Try: 08-04: 19:56

# In[191]:


def assess_fit_normal(data):
    kstest_result = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_lognormal_before1048(data):
    log_data = np.log(data)
    kstest_result = stats.kstest(log_data, 'norm')
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_lognormal(data):
    # Lognormal requires positive data
    if np.any(data <= 0):
        return False
    # Fit lognormal parameters
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    # Perform K-S test using these parameters
    kstest_result = stats.kstest(data, 'lognorm', args=(shape, loc, scale))
    return kstest_result.pvalue > 0.05  # Assume a significance level of 0.05

def assess_fit_gamma(data):
    # Estimate gamma parameters
    alpha, loc, beta = stats.gamma.fit(data)
    kstest_result = stats.kstest(data, 'gamma', args=(alpha, loc, beta))
    return kstest_result.pvalue > 0.05

def assess_fit_weibull(data):
    # Estimate Weibull parameters
    c, loc, scale = stats.weibull_min.fit(data)
    kstest_result = stats.kstest(data, 'weibull_min', args=(c, loc, scale))
    return kstest_result.pvalue > 0.05


# ##### Storing it: 08-04: 22:40 - Success

# In[193]:


def evaluate_survey(df, fit_function):
    fit_results = {
        'fits': [],
        'non_fits': []
    }
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        sample_size = len(group['salary_norm'])
        fit = fit_function(group['salary_norm'])
        if fit:
            fit_results['fits'].append((name, sample_size))
        else:
            fit_results['non_fits'].append((name, sample_size))
    return fit_results

surveys = {'df_it': df_it_w_l, 'df_k': df_k_w_l, 'df_ai': df_ai_w_l}
distributions = {
    'normal': assess_fit_normal,
    'lognormal': assess_fit_lognormal,
    'gamma': assess_fit_gamma,
    'weibull': assess_fit_weibull
}

results_list = []

for survey_name, df in surveys.items():
    for dist_name, fit_function in distributions.items():
        fit_results = evaluate_survey(df, fit_function)
        fits = fit_results['fits']
        non_fits = fit_results['non_fits']
        total_groups = len(fits) + len(non_fits)
        fit_percentage = (len(fits) / total_groups) * 100 if total_groups > 0 else 0
        
        results_list.append({
            'Survey': survey_name,
            'Distribution': dist_name,
            'Total group count': total_groups,
            'Fit group count': len(fits),
            'Fit Percentage': fit_percentage
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results_list)

# Sort the DataFrame by Fit Percentage in descending order
results_df = results_df.sort_values('Fit Percentage', ascending=False)

# Reorder columns to put 'Total group count' and 'Fit group count' above 'Fit Percentage'
column_order = ['Survey', 'Distribution', 'Total group count', 'Fit group count', 'Fit Percentage']
results_df = results_df[column_order]
results_df


# In[194]:


def evaluate_survey(df, fit_function):
    fit_results = {
        'fits': [],
        'non_fits': []
    }
    all_groups = []
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)
    for name, group in grouped:
        sample_size = len(group['salary_norm'])
        fit = fit_function(group['salary_norm'])
        if fit:
            fit_results['fits'].append((name, sample_size))
            all_groups.append((name, sample_size, 'Fit'))
        else:
            fit_results['non_fits'].append((name, sample_size))
            all_groups.append((name, sample_size, 'Non-Fit'))
    return fit_results, all_groups

surveys = {'df_it': df_it_w_l, 'df_k': df_k_w_l, 'df_ai': df_ai_w_l}
distributions = {
    'normal': assess_fit_normal,
    'lognormal': assess_fit_lognormal,
    'gamma': assess_fit_gamma,
    'weibull': assess_fit_weibull
}

results_list = []
all_groups_list = []

for survey_name, df in surveys.items():
    for dist_name, fit_function in distributions.items():
        fit_results, groups = evaluate_survey(df, fit_function)
        fits = fit_results['fits']
        non_fits = fit_results['non_fits']
        total_groups = len(fits) + len(non_fits)
        fit_percentage = (len(fits) / total_groups) * 100 if total_groups > 0 else 0
        
        results_list.append({
            'Survey': survey_name,
            'Distribution': dist_name,
            'Total group count': total_groups,
            'Fit group count': len(fits),
            'Fit Percentage': fit_percentage
        })
        
        for group in groups:
            all_groups_list.append({
                'Survey': survey_name,
                'Distribution': dist_name,
                'Job Category': group[0][0],
                'Seniority Level': group[0][1],
                'Sample Size': group[1],
                'Fit Status': group[2]
            })

# Create DataFrames from the results
results_df = pd.DataFrame(results_list)
all_groups_df = pd.DataFrame(all_groups_list)

# Sort the results DataFrame by Fit Percentage in descending order
results_df = results_df.sort_values('Fit Percentage', ascending=False)

# Reorder columns
column_order = ['Survey', 'Distribution', 'Total group count', 'Fit group count', 'Fit Percentage']
results_df = results_df[column_order]

# Display the results
#results_df

# Optionally, you can save these DataFrames to CSV files
# results_df.to_csv('distribution_fit_results.csv', index=False)
# all_groups_df.to_csv('all_factorial_groups.csv', index=False)


# In[195]:


results_df.sort_values('Survey')


# In[196]:


# Show all groups that fit the normal distribution for the IT survey
all_groups_df[(all_groups_df['Survey'] == 'df_ai') & 
                    (all_groups_df['Distribution'] == 'lognormal') #& 
                    #(all_groups_df['Fit Status'] == 'Fit')
             ]


# Lognormal is a good fit, maybe even the best fit. Kolmogorov-Smirnov however is not proper to test normality, since it should be used to compare datasets with a preexisting concept of a distribution, and not to compare the data with a distribution generated from the data.

# # Normality

# In[199]:


from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import skew, kurtosis, normaltest


# In[200]:


job_category='Data Scientist/ ML Engineer'
seniority_level='junior'
#country='it'

len(
     df_ai[
    (df_ai['job_category'] == job_category) & 
    (df_ai['seniority_level'] == seniority_level) #&
    #(df_ai['country'] == country) #&
    #(df_ai['year'] == year)
    ]
)


# In[201]:


def lilliefors_normality_test(df):
    # Initialize an empty list to store results
    results = []

    # Group the DataFrame by 'job_category' and 'seniority_level'
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)

    # Iterate over each group
    for (job_category, seniority_level), group in grouped:
        # Extract the 'salary' data, dropping NaN values
        salary_data = group['salary_norm_2024_log'].dropna()

        # Only perform the test if there are enough data points
        if len(salary_data) >= 5:
            # Perform the Lilliefors test
            stat, p_value = lilliefors(salary_data, dist='norm')
        else:
            p_value = None  # Not enough data to perform the test

        # Determine significance levels
        sig_p10 = '*' if p_value is not None and p_value < 0.10 else ''
        sig_p05 = '*' if p_value is not None and p_value < 0.05 else ''
        sig_p01 = '*' if p_value is not None and p_value < 0.01 else ''

        # Create a cell ID
        cell_id = f"{job_category}_{seniority_level}"

        # Append the results to the list
        results.append({
            'cell_id': cell_id,
            'job_category': job_category,
            'seniority_level': seniority_level,
            'p_value': p_value,
            'significance_p<0.10': sig_p10,
            'significance_p<0.05': sig_p05,
            'significance_p<0.01': sig_p01
        })

    # Convert the results list into a DataFrame
    result_df = pd.DataFrame(results)

    return result_df

# Example usage:
# Assuming you have a DataFrame 'df_survey' for a particular survey
result_df = lilliefors_normality_test(df_k_w_l)
result_df


# In[202]:


def lilliefors_normality_test(df):
    # Initialize an empty list to store results
    results = []

    # Group the DataFrame by 'job_category' and 'seniority_level'
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)

    # Iterate over each group
    for (job_category, seniority_level), group in grouped:
        # Extract the 'salary' data, dropping NaN values
        salary_data = group['salary_norm_2024_log'].dropna()

        # Calculate the population (number of data points in the group)
        population = len(salary_data)

        # Only perform the test if there are enough data points
        if population >= 5:
            # Perform the Lilliefors test
            stat, p_value = lilliefors(salary_data, dist='norm')
        else:
            p_value = None  # Not enough data to perform the test

        # Determine significance levels
        sig_p10 = '*' if p_value is not None and p_value < 0.10 else ''
        sig_p05 = '*' if p_value is not None and p_value < 0.05 else ''
        sig_p01 = '*' if p_value is not None and p_value < 0.01 else ''

        # Create a cell ID
        cell_id = f"{job_category}_{seniority_level}"

        # Append the results to the list
        results.append({
            'cell_id': cell_id,
            'job_category': job_category,
            'seniority_level': seniority_level,
            'population': population,
            'p_value': p_value,
            'significance_p<0.10': sig_p10,
            'significance_p<0.05': sig_p05,
            'significance_p<0.01': sig_p01
        })

    # Convert the results list into a DataFrame
    result_df = pd.DataFrame(results)

    return result_df

# Example usage:
# Assuming you have a DataFrame 'df_survey' for a particular survey
result_df = lilliefors_normality_test(df_ai_w_l)
result_df


# In[203]:


def lilliefors_normality_test(df):
    # Initialize an empty list to store results
    results = []

    # Group the DataFrame by 'job_category' and 'seniority_level'
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)

    # Iterate over each group
    for (job_category, seniority_level), group in grouped:
        # Extract the 'salary' data, dropping NaN values
        salary_data = group['salary_norm_2024_log'].dropna()

        # Calculate the population (number of data points in the group)
        population = len(salary_data)

        # Only perform the test if there are enough data points
        if population >= 5:
            # Perform the Lilliefors test
            stat, p_value = lilliefors(salary_data, dist='norm')
            # Calculate Skewness and Kurtosis
            skewness = skew(salary_data)
            kurt = kurtosis(salary_data, fisher=True)  # Excess kurtosis (normal distribution has kurtosis 0)
        else:
            p_value = None  # Not enough data to perform the test
            skewness = None
            kurt = None

        # Determine significance levels
        sig_p10 = '*' if p_value is not None and p_value < 0.10 else ''
        sig_p05 = '*' if p_value is not None and p_value < 0.05 else ''
        sig_p01 = '*' if p_value is not None and p_value < 0.01 else ''

        # Create a cell ID
        cell_id = f"{job_category}_{seniority_level}"

        # Append the results to the list
        results.append({
            'cell_id': cell_id,
            'job_category': job_category,
            'seniority_level': seniority_level,
            'population': population,
            'p_value': p_value,
            'skewness': skewness,
            'kurtosis': kurt,
            'significance_p<0.10': sig_p10,
            'significance_p<0.05': sig_p05,
            'significance_p<0.01': sig_p01
        })

    # Convert the results list into a DataFrame
    result_df = pd.DataFrame(results)

    return result_df

# Example usage:
# Assuming you have a DataFrame 'df_survey' for a particular survey
result_df = lilliefors_normality_test(df_ai_w_l)
result_df


# In[204]:


def normality_tests(df):
    # Initialize an empty list to store results
    results = []

    # Group the DataFrame by 'job_category' and 'seniority_level'
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)

    # Iterate over each group
    for (job_category, seniority_level), group in grouped:
        # Extract the 'salary' data, dropping NaN values
        salary_data = group['salary_norm_2024_log'].dropna()

        # Calculate the population (number of data points in the group)
        population = len(salary_data)

        # Only perform the tests if there are enough data points
        if population >= 5:
            # Perform the Lilliefors test
            stat_l, p_value_l = lilliefors(salary_data, dist='norm')
            # Perform D'Agostino's K-squared test
            stat_k, p_value_k = normaltest(salary_data)
            # Calculate Skewness and Kurtosis
            skewness = skew(salary_data)
            kurt = kurtosis(salary_data, fisher=True)  # Excess kurtosis (normal distribution has kurtosis 0)
        else:
            p_value_l = None
            p_value_k = None
            skewness = None
            kurt = None

        # Determine significance levels for Lilliefors test
        L_sig_p10 = '*' if p_value_l is not None and p_value_l < 0.10 else ''
        L_sig_p05 = '*' if p_value_l is not None and p_value_l < 0.05 else ''
        L_sig_p01 = '*' if p_value_l is not None and p_value_l < 0.01 else ''

        # Determine significance levels for D'Agostino's K-squared test
        K_sig_p10 = '*' if p_value_k is not None and p_value_k < 0.10 else ''
        K_sig_p05 = '*' if p_value_k is not None and p_value_k < 0.05 else ''
        K_sig_p01 = '*' if p_value_k is not None and p_value_k < 0.01 else ''

        # Append the results to the list
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'population': population,
            'L_p': p_value_l,
            'L_sig_p<0.10': L_sig_p10,
            'L_sig_p<0.05': L_sig_p05,
            'L_sig_p<0.01': L_sig_p01,
            'skewness': skewness,
            'kurtosis': kurt,
            'K_p': p_value_k,
            'K_sig_p<0.10': K_sig_p10,
            'K_sig_p<0.05': K_sig_p05,
            'K_sig_p<0.01': K_sig_p01
        })

    # Convert the results list into a DataFrame
    result_df = pd.DataFrame(results)

    # Rearranging the columns as specified
    result_df = result_df[
        ['job_category', 'seniority_level', 'population',
         'L_p', 'L_sig_p<0.10', 'L_sig_p<0.05', 'L_sig_p<0.01',
         'skewness', 'kurtosis',
         'K_p', 'K_sig_p<0.10', 'K_sig_p<0.05', 'K_sig_p<0.01']
    ]

    return result_df

# Example usage:
# Assuming you have a DataFrame 'df_survey' for a particular survey
result_df = normality_tests(df_k_w_l)
result_df


# In[205]:


def normality_tests(df):
    # Initialize an empty list to store results
    results = []

    # Group the DataFrame by 'job_category' and 'seniority_level'
    grouped = df.groupby(['job_category', 'seniority_level'], observed=True)

    # Iterate over each group
    for (job_category, seniority_level), group in grouped:
        # Extract the 'salary' data, dropping NaN values
        salary_data = group['salary_norm_2024_log'].dropna()

        # Calculate the population (number of data points in the group)
        population = len(salary_data)

        # Only perform the tests if there are enough data points
        if population >= 5:
            # Perform the Lilliefors test
            stat_l, p_value_l = lilliefors(salary_data, dist='norm')
            # Perform D'Agostino's K-squared test
            stat_k, p_value_k = normaltest(salary_data)
            # Calculate Skewness and Kurtosis
            skewness = skew(salary_data)
            kurt = kurtosis(salary_data, fisher=True)  # Excess kurtosis (normal distribution has kurtosis 0)
        else:
            p_value_l = None
            p_value_k = None
            skewness = None
            kurt = None

        # Determine significance levels for Lilliefors test
        L_sig_p10 = '*' if p_value_l is not None and p_value_l < 0.10 else ''
        L_sig_p05 = '*' if p_value_l is not None and p_value_l < 0.05 else ''
        L_sig_p01 = '*' if p_value_l is not None and p_value_l < 0.01 else ''

        # Determine significance levels for D'Agostino's K-squared test
        K_sig_p10 = '*' if p_value_k is not None and p_value_k < 0.10 else ''
        K_sig_p05 = '*' if p_value_k is not None and p_value_k < 0.05 else ''
        K_sig_p01 = '*' if p_value_k is not None and p_value_k < 0.01 else ''

        # Append the results to the list
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'population': population,
            'L_p': p_value_l,
            'L_sig_p<0.10': L_sig_p10,
            'L_sig_p<0.05': L_sig_p05,
            'L_sig_p<0.01': L_sig_p01,
            'skewness': skewness,
            'kurtosis': kurt,
            'K_p': p_value_k,
            'K_sig_p<0.10': K_sig_p10,
            'K_sig_p<0.05': K_sig_p05,
            'K_sig_p<0.01': K_sig_p01
        })

    # Convert the results list into a DataFrame
    result_df = pd.DataFrame(results)

    # Rearranging the columns as specified
    result_df = result_df[
        ['job_category', 'seniority_level', 'population',
         'L_p', 'L_sig_p<0.10', 'L_sig_p<0.05', 'L_sig_p<0.01',
         'skewness', 'kurtosis',
         'K_p', 'K_sig_p<0.10', 'K_sig_p<0.05', 'K_sig_p<0.01']
    ]

    return result_df

# Example usage:
# Assuming you have a DataFrame 'df_survey' for a particular survey
result_df = normality_tests(df_it_w_l)
result_df


# In[206]:


def assess_practical_normality(result_df, skewness_thr_small=0.5, kurtosis_thr_small=1, 
                               skewness_thr_medium=1, kurtosis_thr_medium=2, 
                               skewness_thr_large=1.5, kurtosis_thr_large=3):
    """
    This function assesses practical normality based on skewness, kurtosis, and population size.
    It adds a new column 'practical_normality' to the input result_df.
    
    Parameters:
    result_df (pd.DataFrame): DataFrame containing the normality test results.
    skewness_thr_small, kurtosis_thr_small: Thresholds for small samples (n < 50)
    skewness_thr_medium, kurtosis_thr_medium: Thresholds for medium samples (50 <= n <= 300)
    skewness_thr_large, kurtosis_thr_large: Thresholds for large samples (n > 300)
    
    Returns:
    pd.DataFrame: Modified DataFrame with a new column 'practical_normality' filled with 'yes' or 'no'.
    """
    
    # Initialize the new column for practical normality assessment
    result_df['practical_normality'] = 'no'  # Default to 'no', we'll set it to 'yes' when conditions are met

    # Iterate through each row in the DataFrame
    for index, row in result_df.iterrows():
        population = row['population']
        skewness = row['skewness']
        kurt = row['kurtosis']

        # Small sample size: n < 50
        if population < 50:
            if abs(skewness) <= skewness_thr_small and abs(kurt) <= kurtosis_thr_small:
                result_df.at[index, 'practical_normality'] = 'yes'

        # Medium sample size: 50 <= n <= 300
        elif 50 <= population <= 300:
            if abs(skewness) <= skewness_thr_medium and abs(kurt) <= kurtosis_thr_medium:
                result_df.at[index, 'practical_normality'] = 'yes'

        # Large sample size: n > 300
        else:
            if abs(skewness) <= skewness_thr_large and abs(kurt) <= kurtosis_thr_large:
                result_df.at[index, 'practical_normality'] = 'yes'
    
    return result_df

# Example usage
# Assuming 'result_df' is your DataFrame from the normality test
result_df = assess_practical_normality(result_df)
result_df[['job_category', 'seniority_level', 'population', 'skewness', 'kurtosis', 'practical_normality']]


# In[207]:


# Total population across all factorial cells
total_population = result_df['population'].sum()
# Population of cells that are considered practically normal
practical_normal_population = result_df[result_df['practical_normality'] == 'yes']['population'].sum()
# Calculate the proportion of practically normal cells weighted by population
practical_significant_proportion = (practical_normal_population / total_population) * 100
print(f"Practical significant normal proportion of the dataframe: {practical_significant_proportion:.2f}%")


# # Independence

# Yearly repetition could make it a paired setup, if the same individuals answered the survey each year.\
# The same individuals may participate in multiple surveys and/or across multiple years, leading to correlated observations.
# I assume independence, but it's likely a mix of the two.
# 
# Impact on Analysis:
# Underestimation of Standard Errors: Ignoring the correlation between repeated measures can lead to underestimating standard errors, increasing the risk of Type I errors (false positives).

# # Levene: Homogeneity of variances (between surveys) (Levene)

# ## Pooled Approach

# In[212]:


# Combine dataframes
df_conc = pd.concat([df_ai, df_it, df_k])

# Define a function to perform Levene's test for each factorial group
def analyze_factorial_groups(df):
    # Get unique factorial groups
    factorial_groups = df.groupby(['job_category', 'seniority_level'], observed=True).size().reset_index(name='count')
    
    results = []
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df = df[
            (df['job_category'] == job_category) &
            (df['seniority_level'] == seniority_level)
        ]
        
        # Check if there are at least 2 surveys with data in this group
        if subset_df['survey'].nunique() < 2:
            continue
        
        # Prepare data for Levene's test
        survey_groups = [subset_df[subset_df['survey'] == survey]['salary_norm_2024'] for survey in subset_df['survey'].unique()]
        
        # Levene's test for homogeneity of variances
        stat, p_value_levene = stats.levene(*survey_groups)
        
        # Determine significance for various levels
        significance_0_10 = '*' if p_value_levene < 0.10 else ''
        significance_0_05 = '*' if p_value_levene < 0.05 else ''
        significance_0_01 = '*' if p_value_levene < 0.01 else ''
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Levene_p_value': p_value_levene,
            'Significance_p<0.10': significance_0_10,
            'Significance_p<0.05': significance_0_05,
            'Significance_p<0.01': significance_0_01
        })
    
    return pd.DataFrame(results)

# Run the analysis for each factorial group
results_df = analyze_factorial_groups(df_conc)

# Output results
print('Levene\'s Test results:')
print('Null Hypothesis: The variances across different surveys are equal (homoscedasticity). In other words, it assumes that the variability in salaries is similar across surveys within each factorial group.')
print('Significance at different p-levels indicates whether the null hypothesis can be rejected. If the p-value is below a significance level, it suggests that the variances differ significantly between surveys, which may impact the validity of combining the surveys for further analysis.')
results_df

# Optionally, save results to a CSV file
#results_df.to_csv('levene_results.csv', index=False)


# ## Pairwise

# In[214]:


# Define a function to perform Levene's test for each factorial group
def analyze_factorial_groups_pairwise(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_conc = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_conc.groupby(['job_category', 'seniority_level'], observed=True).size().reset_index(name='count')
    
    results = []
    homoscedastic_groups = 0
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for Levene's test
        survey_groups = [subset_df1['salary_norm_2024'], subset_df2['salary_norm_2024']]
        
        # Levene's test for homogeneity of variances
        stat, p_value_levene = stats.levene(*survey_groups)
        
        # Check if variances are homoscedastic (p-value >= 0.05)
        is_homoscedastic = p_value_levene >= 0.05
        if is_homoscedastic:
            homoscedastic_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Levene_p_value': p_value_levene,
            'Is_Homoscedastic': is_homoscedastic
        })
    
    # Calculate percentage of homoscedastic groups
    total_groups = len(results)
    homoscedastic_percentage = (homoscedastic_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), homoscedastic_percentage

# Run the analysis for each pair of dataframes
results_k_ai, percentage_k_ai = analyze_factorial_groups_pairwise(df_k, df_ai, 'df_k', 'df_ai')
results_k_it, percentage_k_it = analyze_factorial_groups_pairwise(df_k, df_it, 'df_k', 'df_it')
results_ai_it, percentage_ai_it = analyze_factorial_groups_pairwise(df_ai, df_it, 'df_ai', 'df_it')

# Print the percentage of homoscedastic groups for each pair
print(f'Percentage of homoscedastic groups (df_k vs. df_ai): {percentage_k_ai:.2f}%')
print(f'Percentage of homoscedastic groups (df_k vs. df_it): {percentage_k_it:.2f}%')
print(f'Percentage of homoscedastic groups (df_ai vs. df_it): {percentage_ai_it:.2f}%')

# Optionally, save results to CSV files
# results_k_ai.to_csv('levene_results_k_ai.csv', index=False)
# results_k_it.to_csv('levene_results_k_it.csv', index=False)
# results_ai_it.to_csv('levene_results_ai_it.csv', index=False)


# In[215]:


# Survey labels
labels = ['Kaggle', 'AI-Jobs.net', 'DE IT-survey']

# Coordinates for the triangle vertices (equilateral triangle)
triangle_coords = np.array([
    [0, 1],   # df_k
    [-0.87, -0.5],  # df_ai
    [0.87, -0.5]   # df_it
])

# Create a plot
plt.figure(figsize=(8, 8))
ax = plt.gca()

# Plot the nodes (surveys)
for i, coord in enumerate(triangle_coords):
    ax.text(coord[0], coord[1], labels[i], fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1.5'))

# Define the pairs and corresponding percentages
pairs = [(0, 1, percentage_k_ai), (0, 2, percentage_k_it), (1, 2, percentage_ai_it)]

# Draw shortened arrows between the surveys with the percentage labels
for start, end, percentage in pairs:
    # Calculate the shortened start and end points
    start_x, start_y = triangle_coords[start]
    end_x, end_y = triangle_coords[end]
    
    # Calculate direction vector and shorten the arrow by 15%
    direction_x = end_x - start_x
    direction_y = end_y - start_y
    arrow_length = 0.20  # Shortening factor
    
    start_shortened_x = start_x + arrow_length * direction_x
    start_shortened_y = start_y + arrow_length * direction_y
    end_shortened_x = end_x - arrow_length * direction_x
    end_shortened_y = end_y - arrow_length * direction_y
    
    # Create the arrow with heads on both ends
    ax.annotate(
        '', xy=(end_shortened_x, end_shortened_y), xytext=(start_shortened_x, start_shortened_y),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
    )
    
    # Calculate the midpoint for placing the percentage text
    mid_x = (start_shortened_x + end_shortened_x) / 2
    mid_y = (start_shortened_y + end_shortened_y) / 2
    
    # Offset the text to avoid obstruction by the arrow
    offset_x = (end_shortened_y - start_shortened_y) * 0.1
    offset_y = (start_shortened_x - end_shortened_x) * 0.1
    
    ax.text(mid_x + offset_x, mid_y + offset_y, f'{percentage:.1f}%', fontsize=12, ha='center', va='center')

# Adjust plot limits and aspect ratio
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
plt.title('Pairwise Homoscedasticity Percentages Between Surveys', fontsize=16)
plt.text(0.5, 0.1, '(Higher percentage = more Homogeneity of variances)', fontsize=12, fontstyle='italic', ha='center', transform=ax.transAxes)

plt.gcf().tight_layout(pad=10)

# Display the plot
plt.show()


# ### Pairwise with Bonferroni-correction

# In[217]:


# Define a function to perform Levene's test for each factorial group with Bonferroni correction
def analyze_factorial_groups_pairwise(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_conc = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_conc.groupby(['job_category', 'seniority_level'], observed=True).size().reset_index(name='count')
    
    results = []
    homoscedastic_groups = 0
    total_tests = len(factorial_groups)  # Total number of comparisons
    
    # Bonferroni corrected alpha level
    alpha = 0.05
    corrected_alpha = alpha / total_tests if total_tests > 0 else np.nan
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for Levene's test
        survey_groups = [subset_df1['salary_norm_2024'], subset_df2['salary_norm_2024']]
        
        # Levene's test for homogeneity of variances
        stat, p_value_levene = stats.levene(*survey_groups)
        
        # Apply Bonferroni correction
        is_homoscedastic = p_value_levene >= corrected_alpha
        if is_homoscedastic:
            homoscedastic_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Levene_p_value': p_value_levene,
            'Corrected_Alpha': corrected_alpha,
            'Is_Homoscedastic': is_homoscedastic
        })
    
    # Calculate percentage of homoscedastic groups
    total_groups = len(results)
    homoscedastic_percentage = (homoscedastic_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), homoscedastic_percentage

# Run the analysis for each pair of dataframes
results_k_ai, percentage_k_ai = analyze_factorial_groups_pairwise(df_k, df_ai, 'df_k', 'df_ai')
results_k_it, percentage_k_it = analyze_factorial_groups_pairwise(df_k, df_it, 'df_k', 'df_it')
results_ai_it, percentage_ai_it = analyze_factorial_groups_pairwise(df_ai, df_it, 'df_ai', 'df_it')

# Print the percentage of homoscedastic groups for each pair
print(f'Percentage of homoscedastic groups (df_k vs. df_ai): {percentage_k_ai:.2f}%')
print(f'Percentage of homoscedastic groups (df_k vs. df_it): {percentage_k_it:.2f}%')
print(f'Percentage of homoscedastic groups (df_ai vs. df_it): {percentage_ai_it:.2f}%')


# In[218]:


# Survey labels
labels = ['Kaggle', 'AI-Jobs.net', 'DE IT-survey']

# Coordinates for the triangle vertices (equilateral triangle)
triangle_coords = np.array([
    [0, 1],   # df_k
    [-0.87, -0.5],  # df_ai
    [0.87, -0.5]   # df_it
])

# Create a plot
plt.figure(figsize=(8, 8))
ax = plt.gca()

# Plot the nodes (surveys)
for i, coord in enumerate(triangle_coords):
    ax.text(coord[0], coord[1], labels[i], fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1.5'))

# Define the pairs and corresponding percentages
pairs = [(0, 1, percentage_k_ai), (0, 2, percentage_k_it), (1, 2, percentage_ai_it)]

# Draw shortened arrows between the surveys with the percentage labels
for start, end, percentage in pairs:
    # Calculate the shortened start and end points
    start_x, start_y = triangle_coords[start]
    end_x, end_y = triangle_coords[end]
    
    # Calculate direction vector and shorten the arrow by 15%
    direction_x = end_x - start_x
    direction_y = end_y - start_y
    arrow_length = 0.20  # Shortening factor
    
    start_shortened_x = start_x + arrow_length * direction_x
    start_shortened_y = start_y + arrow_length * direction_y
    end_shortened_x = end_x - arrow_length * direction_x
    end_shortened_y = end_y - arrow_length * direction_y
    
    # Create the arrow with heads on both ends
    ax.annotate(
        '', xy=(end_shortened_x, end_shortened_y), xytext=(start_shortened_x, start_shortened_y),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
    )
    
    # Calculate the midpoint for placing the percentage text
    mid_x = (start_shortened_x + end_shortened_x) / 2
    mid_y = (start_shortened_y + end_shortened_y) / 2
    
    # Offset the text to avoid obstruction by the arrow
    offset_x = (end_shortened_y - start_shortened_y) * 0.1
    offset_y = (start_shortened_x - end_shortened_x) * 0.1
    
    ax.text(mid_x + offset_x, mid_y + offset_y, f'{percentage:.1f}%', fontsize=12, ha='center', va='center')

# Adjust plot limits and aspect ratio
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
plt.title('Pairwise Homoscedasticity Percentages Between Surveys', fontsize=16)
plt.text(0.5, 0.1, '(Higher percentage = more Homogeneity of variances)', fontsize=12, fontstyle='italic', ha='center', transform=ax.transAxes)

plt.gcf().tight_layout(pad=10)

# Display the plot
plt.show()


# ### Brown-Forscythe without correction

# ### Brown-Forscythe with Bonferroni-correction

# In[221]:


# Define a function to perform the Brown-Forsythe test for each factorial group with Bonferroni correction
def analyze_factorial_groups_pairwise(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_conc = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_conc.groupby(['job_category', 'seniority_level'],observed=True).size().reset_index(name='count')
    
    results = []
    homoscedastic_groups = 0
    total_tests = len(factorial_groups)  # Total number of comparisons
    
    # Bonferroni corrected alpha level
    alpha = 0.05
    corrected_alpha = alpha / total_tests if total_tests > 0 else np.nan
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for Brown-Forsythe test (which is a variant of Levene's test using medians)
        survey_groups = [subset_df1['salary_norm_2024'], subset_df2['salary_norm_2024']]
        
        # Brown-Forsythe test for homogeneity of variances (using the median)
        stat, p_value_bf = stats.levene(*survey_groups, center='median')
        
        # Apply Bonferroni correction
        is_homoscedastic = p_value_bf >= corrected_alpha
        if is_homoscedastic:
            homoscedastic_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Brown_Forsythe_p_value': p_value_bf,
            'Corrected_Alpha': corrected_alpha,
            'Is_Homoscedastic': is_homoscedastic
        })
    
    # Calculate percentage of homoscedastic groups
    total_groups = len(results)
    homoscedastic_percentage = (homoscedastic_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), homoscedastic_percentage

# Run the analysis for each pair of dataframes
results_k_ai, percentage_k_ai = analyze_factorial_groups_pairwise(df_k, df_ai, 'df_k', 'df_ai')
results_k_it, percentage_k_it = analyze_factorial_groups_pairwise(df_k, df_it, 'df_k', 'df_it')
results_ai_it, percentage_ai_it = analyze_factorial_groups_pairwise(df_ai, df_it, 'df_ai', 'df_it')

# Print the percentage of homoscedastic groups for each pair
print(f'Percentage of homoscedastic groups (df_k vs. df_ai): {percentage_k_ai:.2f}%')
print(f'Percentage of homoscedastic groups (df_k vs. df_it): {percentage_k_it:.2f}%')
print(f'Percentage of homoscedastic groups (df_ai vs. df_it): {percentage_ai_it:.2f}%')


# ### Practical significance

# In[223]:


df_name = df_combined

# Initialize a list to store the variance ratios
variance_ratios_list = []

# Get unique combinations of 'job_category' and 'seniority_level'
factorial_cells = df_combined.groupby(['job_category', 'seniority_level'], observed=True).size().reset_index().iloc[:, :2]
surveys = df_combined['survey'].unique()

# Calculate variance ratios for each factorial cell
for _, row in factorial_cells.iterrows():
    job_category = row['job_category']
    seniority_level = row['seniority_level']
    cell_data = df_combined[(df_combined['job_category'] == job_category) & (df_combined['seniority_level'] == seniority_level)]
    cell_surveys = cell_data['survey'].unique()

    if len(cell_surveys) < 2:
        continue  # Skip if fewer than two surveys

    variances = cell_data.groupby('survey', observed=True)['salary_norm_2024_log'].var()
    for i in range(len(surveys)):
        for j in range(i+1, len(surveys)):
            survey_i, survey_j = surveys[i], surveys[j]
            if survey_i in cell_surveys and survey_j in cell_surveys:
                var_i, var_j = variances[survey_i], variances[survey_j]
                variance_ratio = var_i / var_j if var_i > var_j else var_j / var_i
                variance_ratios_list.append({
                    'job_category': job_category,
                    'seniority_level': seniority_level,
                    'survey_pair': f"{survey_i} vs. {survey_j}",
                    'variance_ratio': variance_ratio
                })

# Convert results to DataFrame and pivot for heatmap
variance_ratios_df = pd.DataFrame(variance_ratios_list)
variance_ratios_pivot = variance_ratios_df.pivot_table(
    index=['job_category', 'seniority_level'],
    columns='survey_pair',
    values='variance_ratio'
)

# Plot the heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(variance_ratios_pivot, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Variance Ratio'})
plt.title('Variance Ratios Between Survey Pairs by Job Category and Seniority Level')
plt.xlabel('Survey Pair')
plt.ylabel('Job Category and Seniority Level')
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[224]:


# Optionally, analyze the variance ratios
# For example, count how many variance ratios exceed a threshold
threshold = 2  # You can change this value to 3 if preferred
high_variance_ratios = variance_ratios_df[variance_ratios_df['variance_ratio'] >= threshold]

print(f"\nNumber of comparisons with variance ratio ≥ {threshold}: {len(high_variance_ratios)}")
print(f"Total number of comparisons: {len(variance_ratios_df)}")
print(f"Percentage of high variance ratios: {len(high_variance_ratios) / len(variance_ratios_df) * 100:.2f}%")


# In[225]:


df_name = df_combined

# Step 1: Compute the group medians
df_name['group_median'] = df_name.groupby(['survey', 'job_category', 'seniority_level'], observed=True)['salary_norm_2024_log'].transform('median')

# Step 2: Calculate the absolute deviations from the group medians
df_name['abs_dev'] = np.abs(df_name['salary_norm_2024_log'] - df_name['group_median'])

# Violin plot by survey
plt.figure(figsize=(10, 4))
sns.violinplot(x='survey', y='abs_dev', data=df_name, inner="box", density_norm='width')
plt.title('Distribution of Absolute Deviations from Group Medians by Survey')
plt.xlabel('Survey')
plt.ylabel('Absolute Deviation from Group Median')
plt.ylim(-0.12, 1.75)
plt.show()


# In[226]:


df_name = df_combined

# Step 1: Compute the group medians
df_name['group_median'] = df_name.groupby(['survey', 'job_category', 'seniority_level'], observed=True)['salary_norm_2024_log'].transform('median')

# Step 2: Calculate the absolute deviations from the group medians
df_name['abs_dev'] = np.abs(df_name['salary_norm_2024_log'] - df_name['group_median'])

# Violin plot by survey
plt.figure(figsize=(10, 4))
sns.violinplot(x='seniority_level', y='abs_dev', data=df_name, inner="box", density_norm='width')
plt.title('Distribution of Absolute Deviations from Group Medians by Survey')
plt.xlabel('Survey')
plt.ylabel('Absolute Deviation from Group Median')
plt.ylim(-0.15, 2.75)
plt.show()


# In[227]:


df_name = df_combined

# Step 1: Compute the group medians
df_name['group_median'] = df_name.groupby(['survey', 'job_category', 'seniority_level'], observed=True)['salary_norm_2024_log'].transform('median')

# Step 2: Calculate the absolute deviations from the group medians
df_name['abs_dev'] = np.abs(df_name['salary_norm_2024_log'] - df_name['group_median'])

# Violin plot by survey
plt.figure(figsize=(16, 4))
sns.violinplot(x='job_category', y='abs_dev', data=df_name, inner="box", density_norm='width')
plt.title('Distribution of Absolute Deviations from Group Medians by Survey')
plt.xlabel('Survey')
plt.ylabel('Absolute Deviation from Group Median')
plt.xticks(rotation=45, ha='right')
plt.ylim(-0.15, 2.75)
plt.show()


# # Sensitivity Analysis

# In[229]:


from statsmodels.stats.power import FTestAnovaPower
from scipy.stats.mstats import hmean


# In[230]:


# df_combined = pd.concat([df_ai_w_l, df_it_w_l, df_k_w_l])
# df_k_data = df_k_w_l[df_k_w_l['job_category'].isin(data_fields)]
# df_it_data = df_it_w_l[df_it_w_l['job_category'].isin(data_fields)]
# df_ai_data = df_ai_w_l[df_ai_w_l['job_category'].isin(data_fields)]
# df_data_combined = pd.concat([df_k_data, df_it_data, df_ai_data])


# In[231]:


# Set significance level and desired power
alpha = 0.05
power = 0.80

# List of categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Initialize the power analysis object
power_analysis = FTestAnovaPower()

# Perform sensitivity analysis for each categorical variable
for var in categorical_vars:
    # Number of groups for the variable
    k_groups = df_data_combined[var].nunique()
    
    # Group sizes
    group_sizes = df_data_combined[var].value_counts().values
    
    # Compute harmonic mean of group sizes to adjust for unequal group sizes
    n_per_group = hmean(group_sizes)
    
    # Adjusted total sample size
    nobs = n_per_group * k_groups
    
    # Calculate the smallest detectable effect size (Cohen's f)
    effect_size = power_analysis.solve_power(effect_size=None, nobs=nobs, alpha=alpha, power=power, k_groups=k_groups)
    
    # Transform Cohen's f to eta squared
    eta_squared = effect_size**2 / (effect_size**2 + 1)
    
    print(f"Variable: {var}")
    print(f"  Number of groups (levels): {k_groups}")
    print(f"  Harmonic mean of group sizes: {n_per_group:.2f}")
    print(f"  Adjusted total sample size (nobs): {nobs:.0f}")
    print(f"  Smallest detectable effect size (Cohen's f): {effect_size:.4f}")
    print(f"  Eta squared (partial): {eta_squared:.4f}\n")


# In[232]:


# Set significance level and desired power
alpha = 0.05
power = 0.80

# List of categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Initialize the power analysis object
power_analysis = FTestAnovaPower()

# Perform sensitivity analysis for each categorical variable
for var in categorical_vars:
    # Number of groups for the variable
    k_groups = df_combined[var].nunique()
    
    # Group sizes
    group_sizes = df_combined[var].value_counts().values
    
    # Compute harmonic mean of group sizes to adjust for unequal group sizes
    n_per_group = hmean(group_sizes)
    
    # Adjusted total sample size
    nobs = n_per_group * k_groups
    
    # Calculate the smallest detectable effect size (Cohen's f)
    effect_size = power_analysis.solve_power(effect_size=None, nobs=nobs, alpha=alpha, power=power, k_groups=k_groups)
    
    # Transform Cohen's f to eta squared
    eta_squared = effect_size**2 / (effect_size**2 + 1)
    
    print(f"Variable: {var}")
    print(f"  Number of groups (levels): {k_groups}")
    print(f"  Harmonic mean of group sizes: {n_per_group:.2f}")
    print(f"  Adjusted total sample size (nobs): {nobs:.0f}")
    print(f"  Smallest detectable effect size (Cohen's f): {effect_size:.4f}")
    print(f"  Eta squared (partial): {eta_squared:.4f}\n")

# Sensitivity analysis for the interaction term between 'job_category' and 'seniority_level'
# Create the interaction term
df_combined['interaction'] = df_combined['job_category'].astype(str) + ':' + df_combined['seniority_level'].astype(str)

# Number of groups for the interaction term
k_groups_interaction = df_combined['interaction'].nunique()

# Group sizes for the interaction term
group_sizes_interaction = df_combined['interaction'].value_counts().values

# Compute harmonic mean of group sizes for the interaction term
n_per_group_interaction = hmean(group_sizes_interaction)

# Adjusted total sample size for the interaction term
nobs_interaction = n_per_group_interaction * k_groups_interaction

# Calculate the smallest detectable effect size (Cohen's f) for the interaction term
effect_size_interaction = power_analysis.solve_power(
    effect_size=None,
    nobs=nobs_interaction,
    alpha=alpha,
    power=power,
    k_groups=k_groups_interaction
)

# Transform Cohen's f to eta squared
eta_squared_interaction = effect_size_interaction**2 / (effect_size_interaction**2 + 1)

print("Interaction Term: 'job_category' x 'seniority_level'")
print(f"  Number of groups (levels): {k_groups_interaction}")
print(f"  Harmonic mean of group sizes: {n_per_group_interaction:.2f}")
print(f"  Adjusted total sample size (nobs): {nobs_interaction:.0f}")
print(f"  Smallest detectable effect size (Cohen's f): {effect_size_interaction:.4f}")
print(f"  Eta squared (partial): {eta_squared_interaction:.4f}\n")


# # VIF

# P-Values:	Inflated Type I errors, false significance	Increased Type II errors, unstable significance\
# Partial Eta-Squared:	Inflated for some factors, misleading effect sizes	Underestimated for true effects, arbitrary variance splitting \
# Post-Hoc Power:	Overestimated power, false confidence in significance	Reduced power for true effects, misleading conclusions

# In[235]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# ## Without interaction

# In[237]:


# Define the dataframe to be used
df_name = df_combined

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ### Country

# In[239]:


# Define the dataframe to be used
df_name = df_data_combined

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Define the formula for the model, adding the interaction term
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ## With Interaction

# In[241]:


# Define the dataframe to be used
df_name = df_data_combined

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Define the formula for the model, adding the interaction term
interaction_term = 'C(seniority_level):C(job_category)'
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars]) + ' + ' + interaction_term

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ## IT

# In[243]:


# Define the dataframe to be used
df_name = df_it_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'language_category', 'city','company_size']

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ## AI-jobs

# In[245]:


# Define the dataframe to be used
df_name = df_ai_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'country','company_size']

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# ## Kaggle

# In[247]:


# Define the dataframe to be used
df_name = df_k_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'country','company_size','education_level']

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns

vif.sort_values(by='VIF', ascending=False).head(50)


# # ANOVA

# In[249]:


import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
import statsmodels.stats.api as sms
from statsmodels.stats.api import het_breuschpagan
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import cramervonmises, kurtosis, skew


# ## Pooled approach

# In[251]:


# 1-Way ANOVA

# Combine dataframes
df_combined = pd.concat([df_ai_w_l, df_it_w_l, df_k_w_l])

# Define a function to perform ANOVA for each factorial group
def anova_for_factorial_groups(df):
    # Get unique factorial groups
    factorial_groups = df.groupby(['job_category', 'seniority_level'],observed=True).size().reset_index(name='count')
    
    results = []
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df = df[
            (df['job_category'] == job_category) &
            (df['seniority_level'] == seniority_level)
        ]
        
        # Check if there are at least 2 surveys with data in this group
        if subset_df['survey'].nunique() < 2:
            continue
        
        # Prepare data for ANOVA
        anova_data = [subset_df[subset_df['survey'] == survey]['salary_norm_2024_log'] for survey in subset_df['survey'].unique()]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*anova_data)
        
        # Determine significance for various levels
        significance_0_10 = '*' if p_value < 0.10 else ''
        significance_0_05 = '*' if p_value < 0.05 else ''
        significance_0_01 = '*' if p_value < 0.01 else ''
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'F_statistic': f_stat,
            'p_value': p_value,
            'Significance@ p<0.10': significance_0_10,
            'Significance@ p<0.05': significance_0_05,
            'Significance@ p<0.01': significance_0_01,
        })
    
    return pd.DataFrame(results)

# Run ANOVA for each factorial group
anova_results = anova_for_factorial_groups(df_combined)

# Output results
print('ANOVA results:')
print('Null Hypothesis was that there is no significant difference between the distributions of the different surveys. In other words, it assumes that combining the surveys is reasonable because the distributions are the same.')
print('Sginificance@ different p-levels means that we can reject the Null Hypothsis. This indicates that there is significant evidence suggesting that the salary distributions differ between surveys, and thus combining the surveys may not be reasonable.')
anova_results

# Optionally, save results to a CSV file
#anova_results.to_csv('anova_results.csv', index=False)


# ## Pairwise approach

# In[253]:


# Define a function to perform ANOVA for each factorial group with Bonferroni correction
def analyze_factorial_groups_pairwise_anova(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_combined = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_combined.groupby(['job_category', 'seniority_level'], observed=True).size().reset_index(name='count')
    
    results = []
    significant_groups = 0
    total_tests = len(factorial_groups)  # Total number of comparisons
    
    # Bonferroni corrected alpha level
    alpha = 0.05
    corrected_alpha = alpha / total_tests if total_tests > 0 else np.nan
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for ANOVA
        survey_groups = [subset_df1['salary_norm_2024_log'], subset_df2['salary_norm_2024_log']]
        
        # ANOVA test for difference in means
        stat, p_value_anova = stats.f_oneway(*survey_groups)
        
        # Apply Bonferroni correction
        is_significant = p_value_anova < corrected_alpha
        if is_significant:
            significant_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'ANOVA_p_value': p_value_anova,
            'Corrected_Alpha': corrected_alpha,
            'Is_Significant': is_significant
        })
    
    # Calculate percentage of significant groups
    total_groups = len(results)
    significant_percentage = (significant_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), significant_percentage

# Run the analysis for each pair of dataframes
results_k_ai_anova, percentage_k_ai_anova = analyze_factorial_groups_pairwise_anova(df_k_w_l, df_ai_w_l, 'df_k', 'df_ai')
results_k_it_anova, percentage_k_it_anova = analyze_factorial_groups_pairwise_anova(df_k_w_l, df_it_w_l, 'df_k', 'df_it')
results_ai_it_anova, percentage_ai_it_anova = analyze_factorial_groups_pairwise_anova(df_ai_w_l, df_it_w_l, 'df_ai', 'df_it')

# Print the percentage of significant groups for each pair
print(f'Percentage of significant groups (df_k vs. df_ai): {percentage_k_ai_anova:.2f}%')
print(f'Percentage of significant groups (df_k vs. df_it): {percentage_k_it_anova:.2f}%')
print(f'Percentage of significant groups (df_ai vs. df_it): {percentage_ai_it_anova:.2f}%')


# ### Welch's ANOVA

# In[255]:


# Define a function to perform Welch's ANOVA for each factorial group with Bonferroni correction
def analyze_factorial_groups_pairwise_welch_anova(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_combined = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_combined.groupby(['job_category', 'seniority_level']).size().reset_index(name='count')
    
    results = []
    significant_groups = 0
    total_tests = len(factorial_groups)  # Total number of comparisons
    
    # Bonferroni corrected alpha level
    alpha = 0.05
    corrected_alpha = alpha / total_tests if total_tests > 0 else np.nan
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for Welch's ANOVA
        survey_groups = [subset_df1['salary_norm_2024_log'], subset_df2['salary_norm_2024_log']]
        
        # Welch's ANOVA test for difference in means
        stat, p_value_welch_anova = stats.ttest_ind(subset_df1['salary_norm_2024_log'], subset_df2['salary_norm_2024_log'], equal_var=False)
        
        # Apply Bonferroni correction
        is_significant = p_value_welch_anova < corrected_alpha
        if is_significant:
            significant_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Welch_ANOVA_p_value': p_value_welch_anova,
            'Corrected_Alpha': corrected_alpha,
            'Is_Significant': is_significant
        })
    
    # Calculate percentage of significant groups
    total_groups = len(results)
    significant_percentage = (significant_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), significant_percentage

# Run the analysis for each pair of dataframes
results_k_ai_welch, percentage_k_ai_welch = analyze_factorial_groups_pairwise_welch_anova(df_k_w_l, df_ai_w_l, 'df_k', 'df_ai')
results_k_it_welch, percentage_k_it_welch = analyze_factorial_groups_pairwise_welch_anova(df_k_w_l, df_it_w_l, 'df_k', 'df_it')
results_ai_it_welch, percentage_ai_it_welch = analyze_factorial_groups_pairwise_welch_anova(df_ai_w_l, df_it_w_l, 'df_ai', 'df_it')

# Print the percentage of significant groups for each pair
print(f'Percentage of significant groups (df_k vs. df_ai): {percentage_k_ai_welch:.2f}%')
print(f'Percentage of significant groups (df_k vs. df_it): {percentage_k_it_welch:.2f}%')
print(f'Percentage of significant groups (df_ai vs. df_it): {percentage_ai_it_welch:.2f}%')


# ### Kruskal-Wallis

# In[257]:


# Define a function to perform Kruskal-Wallis test for each factorial group with Bonferroni correction
def analyze_factorial_groups_pairwise_kruskal_wallis(df1, df2, df1_name, df2_name):
    # Combine the two dataframes
    df_combined = pd.concat([df1, df2])
    
    # Get unique factorial groups
    factorial_groups = df_combined.groupby(['job_category', 'seniority_level']).size().reset_index(name='count')
    
    results = []
    significant_groups = 0
    total_tests = len(factorial_groups)  # Total number of comparisons
    
    # Bonferroni corrected alpha level
    alpha = 0.05
    corrected_alpha = alpha / total_tests if total_tests > 0 else np.nan
    
    for _, group in factorial_groups.iterrows():
        job_category = group['job_category']
        seniority_level = group['seniority_level']
        
        # Filter data for the current factorial group
        subset_df1 = df1[
            (df1['job_category'] == job_category) &
            (df1['seniority_level'] == seniority_level)
        ]
        
        subset_df2 = df2[
            (df2['job_category'] == job_category) &
            (df2['seniority_level'] == seniority_level)
        ]
        
        # Check if both surveys have data in this group
        if subset_df1.empty or subset_df2.empty:
            continue
        
        # Prepare data for Kruskal-Wallis test
        survey_groups = [subset_df1['salary_norm_2024_log'], subset_df2['salary_norm_2024_log']]
        
        # Kruskal-Wallis test for difference in distributions
        stat, p_value_kruskal_wallis = stats.kruskal(*survey_groups)
        
        # Apply Bonferroni correction
        is_significant = p_value_kruskal_wallis < corrected_alpha
        if is_significant:
            significant_groups += 1
        
        # Collect results
        results.append({
            'job_category': job_category,
            'seniority_level': seniority_level,
            'Kruskal_Wallis_p_value': p_value_kruskal_wallis,
            'Corrected_Alpha': corrected_alpha,
            'Is_Significant': is_significant
        })
    
    # Calculate percentage of significant groups
    total_groups = len(results)
    significant_percentage = (significant_groups / total_groups) * 100 if total_groups > 0 else np.nan
    
    return pd.DataFrame(results), significant_percentage

# Run the analysis for each pair of dataframes
results_k_ai_kruskal, percentage_k_ai_kruskal = analyze_factorial_groups_pairwise_kruskal_wallis(df_k_w_l, df_ai_w_l, 'df_k', 'df_ai')
results_k_it_kruskal, percentage_k_it_kruskal = analyze_factorial_groups_pairwise_kruskal_wallis(df_k_w_l, df_it_w_l, 'df_k', 'df_it')
results_ai_it_kruskal, percentage_ai_it_kruskal = analyze_factorial_groups_pairwise_kruskal_wallis(df_ai_w_l, df_it_w_l, 'df_ai', 'df_it')

# Print the percentage of significant groups for each pair
print(f'Percentage of significant groups (df_k vs. df_ai): {percentage_k_ai_kruskal:.2f}%')
print(f'Percentage of significant groups (df_k vs. df_it): {percentage_k_it_kruskal:.2f}%')
print(f'Percentage of significant groups (df_ai vs. df_it): {percentage_ai_it_kruskal:.2f}%')


# ## Multi-way approach

# In[259]:


pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[260]:


df_combined = pd.concat([df_ai_w_l, df_k_w_l, df_it_w_l], ignore_index=True)


# In[261]:


df_name = df_data

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[262]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var


# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_combined.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.01

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_combined.groupby(var_name, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_combined.groupby(var_name, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# In[263]:


# Define the dataframe to be used
df_name = df_combined  # You can update df_combined to any other dataframe as needed

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    k_groups = df_name[actual_var].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_name.groupby(actual_var, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_name.groupby(actual_var, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# In[264]:


# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_combined[var] = df_combined[var].astype('category')

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_combined).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# ### With other normalizations

# In[266]:


df_combined.head(1)


# In[267]:


# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_combined[var] = df_combined[var].astype('category')

# Define the formula for the model
formula = 'salary_normse_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_combined).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# ### Try: 20:23 with interaction

# In[269]:


# Define the dataframe to be used
df_name = df_combined

# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model, adding the interaction term
interaction_term = 'C(seniority_level):C(job_category)'
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars]) + ' + ' + interaction_term

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars + [interaction_term]):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    if var == interaction_term:
        k_groups = df_name.groupby(['job_category', 'seniority_level'], observed=True).ngroups
        group_sizes = df_name.groupby(['job_category', 'seniority_level'], observed=True).size().values
    else:
        k_groups = df_name[actual_var].nunique()
        group_sizes = df_name.groupby(actual_var, observed=True).size().values
    # Calculate the effective sample size for this variable
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']]);


# In[270]:


# Get design matrices for VIF calculation
y, X = dmatrices(formula, data=df_name, return_type='dataframe')

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Feature"] = X.columns


# In[271]:


vif.sort_values(by='VIF', ascending=False).head(50)


# In[272]:


pd.crosstab(df_name['seniority_level'], df_name['job_category'])


# In[273]:


df_name.groupby(['seniority_level', 'job_category'], observed=True).size()


# In[274]:


df_name.groupby('job_category').size()


# In[275]:


# Given total standard deviation from the dependent variable in USD
total_std_dev = df_name['salary_2024'].std()

# Calculate the standard deviation explained by each variable
anova_results['Std_Dev_Explained'] = anova_results['Partial_eta_sq'] * total_std_dev

print(f"Total standard deviation of salaries: {total_std_dev:.0f} USD")
print(f"Total explained standard deviation of salaries: {anova_results['Std_Dev_Explained'].sum():.0f} USD")

# Display the updated ANOVA results
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Std_Dev_Explained', 'Power']])


# In[276]:


# Extract residuals from the fitted model
residuals = model.resid

# Plot histogram of residuals
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde=True, bins=40)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[277]:


# Import necessary libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Extract residuals from the fitted model
residuals = model.resid

# Create a Q-Q plot
plt.figure(figsize=(8, 3))
fig = sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[278]:


# Extract residuals from the fitted model
residuals = model.resid

# Perform the Shapiro-Wilk test
shapiro_stat, shapiro_p_value = stats.shapiro(residuals)

# Create a DataFrame to display the results
normality_results = pd.DataFrame({
    'Test Statistic': [shapiro_stat],
    'p-value': [shapiro_p_value]
})


# Determine significance for various levels and add as new columns
normality_results['Significance@ p<0.10'] = ['*' if shapiro_p_value < 0.10 else '']
normality_results['Significance@ p<0.05'] = ['*' if shapiro_p_value < 0.05 else '']
normality_results['Significance@ p<0.01'] = ['*' if shapiro_p_value < 0.01 else '']

# Display the results
display(normality_results)


# In[279]:


# Take a random sample of 5000 residuals
sample_residuals = residuals.sample(300, random_state=42)

# Run Shapiro-Wilk test on the subsample
shapiro_stat, shapiro_p_value = stats.shapiro(sample_residuals)

# Display sample test results
normality_results = pd.DataFrame({
    'Test Statistic': [shapiro_stat],
    'p-value': [shapiro_p_value],
    'Significance@ p<0.10': ['*' if shapiro_p_value < 0.10 else ''],
    'Significance@ p<0.05': ['*' if shapiro_p_value < 0.05 else ''],
    'Significance@ p<0.01': ['*' if shapiro_p_value < 0.01 else '']
})
display(normality_results)


# In[280]:


# Perform the Anderson-Darling test for normality
anderson_result = stats.anderson(residuals, dist='norm')

# Extract significance level results
ad_statistic = anderson_result.statistic
critical_values = anderson_result.critical_values
significance_levels = anderson_result.significance_level

# Display results
normality_results = pd.DataFrame({
    'Test Statistic (A-D)': [ad_statistic],
    'Significance Levels': [f'{significance_levels}'],
    'Critical Values': [f'{critical_values}'],
})
normality_results


# In[281]:


# Perform the Cramér-von Mises test
cvm_result = cramervonmises(residuals, 'norm')

# Create a DataFrame with results
normality_results = pd.DataFrame({
    'Test Statistic (CVM)': [cvm_result.statistic],
    'p-value': [cvm_result.pvalue],
    'Significance@ p<0.10': ['*' if cvm_result.pvalue < 0.10 else ''],
    'Significance@ p<0.05': ['*' if cvm_result.pvalue < 0.05 else ''],
    'Significance@ p<0.01': ['*' if cvm_result.pvalue < 0.01 else '']
})

# Display the results
normality_results


# In[282]:


# Extract residuals from the fitted model
residuals = model.resid

# Perform the Cramér-von Mises test for normality
cvm_result = cramervonmises(residuals, 'norm')

# Calculate additional statistics
population_size = len(residuals)
kurtosis_value = kurtosis(residuals) # Excess kurtosis! Normal distribution should have 0.
skewness_value = skew(residuals)

# Create a DataFrame with results
normality_results = pd.DataFrame({
    'Test Statistic (CVM)': [cvm_result.statistic],
    'p-value': [cvm_result.pvalue],
    'Population Size': [population_size],
    'Kurtosis': [kurtosis_value],
    'Skewness': [skewness_value],
    'Significance@ p<0.10': ['*' if cvm_result.pvalue < 0.10 else ''],
    'Significance@ p<0.05': ['*' if cvm_result.pvalue < 0.05 else ''],
    'Significance@ p<0.01': ['*' if cvm_result.pvalue < 0.01 else '']
})

# Display the results
normality_results


# In[283]:


# Parameters for the simulation
min_sample_size = 20     # Minimum sample size
max_sample_size = 1000   # Maximum sample size
step_size = 1            # Step size for increasing sample size
num_simulations = 20     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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

        # Append individual result to the list
        results.append({'Sample Size': size, 'Shapiro-Wilk p-value': p_value})

# Convert results to a DataFrame for easy plotting
results_df = pd.DataFrame(results)

# Plot the Sample Size vs. Average P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Shapiro-Wilk p-value'], alpha=0.1, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Shapiro-Wilk p-value')
plt.title('Sample Size vs. Shapiro-Wilk p-values (across simulations)')
plt.legend()
plt.show()


# In[284]:


# Parameters for the simulation
min_sample_size = 20     # Minimum sample size
max_sample_size = 1000   # Maximum sample size
step_size = 10           # Step size for increasing sample size
num_simulations = 1000     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Shapiro-Wilk p-value'], alpha=0.5, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Shapiro-Wilk p-value')
plt.title('Sample Size vs. Average Shapiro-Wilk p-value (across simulations)')
plt.legend()
plt.show()


# In[285]:


# Parameters for the simulation
min_sample_size = 10     # Minimum sample size
max_sample_size = 75   # Maximum sample size
step_size = 1           # Step size for increasing sample size
num_simulations = 100     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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
        
        # Perform Cramér-von Mises test and store the p-value
        cvm_result = cramervonmises(subsample, 'norm')
        p_values.append(cvm_result.pvalue)
    
    # Calculate the average p-value across simulations
    avg_p_value = np.mean(p_values)
    
    # Append results to the list
    results.append({
        'Sample Size': size,
        'Average Cramér-von Mises p-value': avg_p_value,
        'Kurtosis': kurtosis(subsample),
        'Skewness': skew(subsample)
    })

# Convert results to a DataFrame for easy plotting
results_df = pd.DataFrame(results)

# Plot the Sample Size vs. Average P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Cramér-von Mises p-value'], alpha=0.5, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Cramér-von Mises p-value')
plt.title('Sample Size vs. Average Cramér-von Mises p-value (across simulations)')
plt.legend()
plt.show()


# ANOVA’s Robustness to Mild Deviations: ANOVA is generally robust to minor deviations from normality, especially with larger sample sizes. With residuals that only start to show significant deviations at around 400 samples, it’s likely that these deviations won’t substantially affect your F-test results or interpretations, particularly given that you have 25,000 observations in total.

# I would consider the residuals to be close enough to normal for ANOVA purposes. The deviations are statistically significant with larger samples, but their practical impact is minimal given the robustness of ANOVA to mild normality violations.

# In[288]:


# Plot residuals to check for homoscedasticity
def plot_residuals(model):
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Scatter plot of residuals vs fitted values
    plt.scatter(fitted_values, residuals, alpha=0.05, s=5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

# Example usage:
plot_residuals(model)


# In[289]:


# Assuming 'model' is your fitted ANOVA or regression model
residuals = model.resid
exog = model.model.exog  # Independent variables used in the model

bp_test = sms.het_breuschpagan(residuals, exog)
print(f"Breusch-Pagan test statistic: {bp_test[0]}, p-value: {bp_test[1]}")


# In[290]:


exog_white = sm.add_constant(np.column_stack([exog, exog**2]))  # Adding squared terms for White's test
white_test = sms.het_breuschpagan(residuals, exog_white)
print(f"White's test statistic: {white_test[0]}, p-value: {white_test[1]}")


# In[291]:


# Group residuals by each factorial cell (survey, job_category, seniority_level)
df_combined['residuals'] = model.resid
groups = [df_combined[df_combined['survey'] == survey]['residuals'].abs() for survey in df_combined['survey'].unique()]

# Brown-Forsythe (modified Levene) test
brown_forsythe = levene(*groups, center='median')
print(f"Brown-Forsythe test statistic: {brown_forsythe.statistic}, p-value: {brown_forsythe.pvalue}")


# In[292]:


# Group residuals by each factorial cell (survey, job_category, seniority_level)
df_combined['residuals'] = model.resid
factorial_cells = df_combined.groupby(['survey', 'job_category', 'seniority_level'])

# Extract absolute residuals for each factorial cell
groups = [cell['residuals'].abs().values for _, cell in factorial_cells]

# Brown-Forsythe (modified Levene) test across all factorial cells
brown_forsythe = levene(*groups, center='median')
print(f"Brown-Forsythe test statistic: {brown_forsythe.statistic}, p-value: {brown_forsythe.pvalue}")


# In[293]:


# Extract residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Parameters for the simulation
min_sample_size = 20      # Minimum sample size
max_sample_size = 17000    # Maximum sample size
step_size = 100            # Step size for increasing sample size
num_simulations = 100     # Number of simulations per sample size
dot_size = 10             # Size of the dots in the plot

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

# Plot the Sample Size vs. Average Breusch-Pagan P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Breusch-Pagan p-value'], alpha=0.5, s=dot_size, c="purple")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Breusch-Pagan p-value')
plt.title('Sample Size vs. Average Breusch-Pagan p-value (across simulations)')
plt.legend()
plt.show()


# #### TODO: Type 3 explanation
# Performs ANOVA with Type III sum of squares, which accounts for the hierarchical structure of the model and the presence of other variables.

# Conclusion:\
# There is statistical significance between the surveys, as indicated by the ANOVA results, and the large sample sizes across factorial cells enhance our confidence in these findings. However, the magnitude of the observed differences, as reflected in the weighted average Eta-squared values, suggests that these differences are relatively small in practical terms.
# 
# The weighted average Eta-squared values indicate that the differences between surveys account for only 0.8%, 2.7%, and 3.1% of the variance in salaries for the respective survey comparisons (df_k vs. df_it, df_k vs. df_ai, and df_ai vs. df_it). This means that while the surveys do exhibit statistically significant differences, the practical impact of these differences is minor. The majority of the variance in salaries is explained by factors other than the survey source, such as seniority level, job category, and other relevant factors (other individual differences between respondants).
# 
# In summary, while the differences between the surveys are statistically detectable, the effect sizes are small, indicating that the surveys are more similar than different in terms of the salary distributions they represent.

# ### Multiway - IT

# In[297]:


df_it_w_l.head(1)


# In[298]:


# Define the dataframe to be used
df_name = df_it_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'language_at_work', 'company_size']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model, adding the interaction term
interaction_term = ''#'C(seniority_level):C(job_category)'
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])# + ' + ' + interaction_term

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars + [interaction_term]):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    if var == interaction_term:
        k_groups = df_name.groupby(['job_category', 'seniority_level'], observed=True).ngroups
        group_sizes = df_name.groupby(['job_category', 'seniority_level'], observed=True).size().values
    else:
        k_groups = df_name[actual_var].nunique()
        group_sizes = df_name.groupby(actual_var, observed=True).size().values
    # Calculate the effective sample size for this variable
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
anova_results


# In[299]:


# Extract residuals from the fitted model
residuals = model.resid

# Plot histogram of residuals
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde=True, bins=100)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[300]:


# Extract residuals from the fitted model
residuals = model.resid

# Create a Q-Q plot
plt.figure(figsize=(8, 3))
fig = sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[301]:


# Extract residuals from the fitted model
residuals = model.resid

# Perform the Cramér-von Mises test for normality
cvm_result = cramervonmises(residuals, 'norm')

# Calculate additional statistics
population_size = len(residuals)
kurtosis_value = kurtosis(residuals) # Excess kurtosis! Normal distribution should have 0.
skewness_value = skew(residuals)

# Create a DataFrame with results
normality_results = pd.DataFrame({
    'Test Statistic (CVM)': [cvm_result.statistic],
    'p-value': [cvm_result.pvalue],
    'Population Size': [population_size],
    'Kurtosis': [kurtosis_value],
    'Skewness': [skewness_value],
    'Significance@ p<0.10': ['*' if cvm_result.pvalue < 0.10 else ''],
    'Significance@ p<0.05': ['*' if cvm_result.pvalue < 0.05 else ''],
    'Significance@ p<0.01': ['*' if cvm_result.pvalue < 0.01 else '']
})

# Display the results
normality_results


# In[302]:


# Parameters for the simulation
min_sample_size = 20     # Minimum sample size
max_sample_size = 1000   # Maximum sample size
step_size = 10           # Step size for increasing sample size
num_simulations = 100     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Shapiro-Wilk p-value'], alpha=0.5, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Shapiro-Wilk p-value')
plt.title('Sample Size vs. Average Shapiro-Wilk p-value (across simulations)')
plt.legend()
plt.show()


# In[303]:


# Plot residuals to check for homoscedasticity
def plot_residuals(model):
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Scatter plot of residuals vs fitted values
    plt.scatter(fitted_values, residuals, alpha=0.1, s=5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

# Example usage:
plot_residuals(model)


# In[304]:


# Extract residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Parameters for the simulation
min_sample_size = 20      # Minimum sample size
max_sample_size = 2000    # Maximum sample size
step_size = 10            # Step size for increasing sample size
num_simulations = 10     # Number of simulations per sample size
dot_size = 10             # Size of the dots in the plot

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

# Plot the Sample Size vs. Average Breusch-Pagan P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Breusch-Pagan p-value'], alpha=0.5, s=dot_size, c="purple")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Breusch-Pagan p-value')
plt.title('Sample Size vs. Average Breusch-Pagan p-value (across simulations)')
plt.legend()
plt.show()


# ### Multiway - AI-Jobs

# In[306]:


df_ai_w_l.head(1)


# In[307]:


# Define the dataframe to be used
df_name = df_ai_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'company_size']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model, adding the interaction term
interaction_term = ''#'C(seniority_level):C(job_category)'
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])# + ' + ' + interaction_term

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars + [interaction_term]):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    if var == interaction_term:
        k_groups = df_name.groupby(['job_category', 'seniority_level'], observed=True).ngroups
        group_sizes = df_name.groupby(['job_category', 'seniority_level'], observed=True).size().values
    else:
        k_groups = df_name[actual_var].nunique()
        group_sizes = df_name.groupby(actual_var, observed=True).size().values
    # Calculate the effective sample size for this variable
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
anova_results


# In[308]:


# Extract residuals from the fitted model
residuals = model.resid

# Plot histogram of residuals
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde=True, bins=40)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[309]:


# Extract residuals from the fitted model
residuals = model.resid

# Create a Q-Q plot
plt.figure(figsize=(8, 3))
fig = sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[310]:


# Extract residuals from the fitted model
residuals = model.resid

# Perform the Cramér-von Mises test for normality
cvm_result = cramervonmises(residuals, 'norm')

# Calculate additional statistics
population_size = len(residuals)
kurtosis_value = kurtosis(residuals) # Excess kurtosis! Normal distribution should have 0.
skewness_value = skew(residuals)

# Create a DataFrame with results
normality_results = pd.DataFrame({
    'Test Statistic (CVM)': [cvm_result.statistic],
    'p-value': [cvm_result.pvalue],
    'Population Size': [population_size],
    'Kurtosis': [kurtosis_value],
    'Skewness': [skewness_value],
    'Significance@ p<0.10': ['*' if cvm_result.pvalue < 0.10 else ''],
    'Significance@ p<0.05': ['*' if cvm_result.pvalue < 0.05 else ''],
    'Significance@ p<0.01': ['*' if cvm_result.pvalue < 0.01 else '']
})

# Display the results
normality_results


# In[311]:


# Parameters for the simulation
min_sample_size = 20     # Minimum sample size
max_sample_size = 5000   # Maximum sample size
step_size = 50           # Step size for increasing sample size
num_simulations = 100     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Shapiro-Wilk p-value'], alpha=0.5, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Shapiro-Wilk p-value')
plt.title('Sample Size vs. Average Shapiro-Wilk p-value (across simulations)')
plt.legend()
plt.show()


# In[312]:


# Plot residuals to check for homoscedasticity
def plot_residuals(model):
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Scatter plot of residuals vs fitted values
    plt.scatter(fitted_values, residuals, alpha=0.05, s=5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

# Example usage:
plot_residuals(model)


# In[313]:


# Extract residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Parameters for the simulation
min_sample_size = 20      # Minimum sample size
max_sample_size = 2000    # Maximum sample size
step_size = 10            # Step size for increasing sample size
num_simulations = 10     # Number of simulations per sample size
dot_size = 10             # Size of the dots in the plot

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

# Plot the Sample Size vs. Average Breusch-Pagan P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Breusch-Pagan p-value'], alpha=0.5, s=dot_size, c="purple")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Breusch-Pagan p-value')
plt.title('Sample Size vs. Average Breusch-Pagan p-value (across simulations)')
plt.legend()
plt.show()


# ### Multiway - Kaggle

# In[315]:


df_k_w_l.head(1)


# In[316]:


# Define the dataframe to be used
df_name = df_k_w_l

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category','company_size','education_level','year']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model, adding the interaction term
interaction_term = ''#'C(seniority_level):C(job_category)'
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])# + ' + ' + interaction_term

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars + [interaction_term]):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    if var == interaction_term:
        k_groups = df_name.groupby(['job_category', 'seniority_level'], observed=True).ngroups
        group_sizes = df_name.groupby(['job_category', 'seniority_level'], observed=True).size().values
    else:
        k_groups = df_name[actual_var].nunique()
        group_sizes = df_name.groupby(actual_var, observed=True).size().values
    # Calculate the effective sample size for this variable
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
anova_results


# In[317]:


# Extract residuals from the fitted model
residuals = model.resid

# Plot histogram of residuals
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde=True, bins=100)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# In[318]:


# Extract residuals from the fitted model
residuals = model.resid

# Create a Q-Q plot
plt.figure(figsize=(8, 3))
fig = sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[319]:


# Extract residuals from the fitted model
residuals = model.resid

# Perform the Cramér-von Mises test for normality
cvm_result = cramervonmises(residuals, 'norm')

# Calculate additional statistics
population_size = len(residuals)
kurtosis_value = kurtosis(residuals) # Excess kurtosis! Normal distribution should have 0.
skewness_value = skew(residuals)

# Create a DataFrame with results
normality_results = pd.DataFrame({
    'Test Statistic (CVM)': [cvm_result.statistic],
    'p-value': [cvm_result.pvalue],
    'Population Size': [population_size],
    'Kurtosis': [kurtosis_value],
    'Skewness': [skewness_value],
    'Significance@ p<0.10': ['*' if cvm_result.pvalue < 0.10 else ''],
    'Significance@ p<0.05': ['*' if cvm_result.pvalue < 0.05 else ''],
    'Significance@ p<0.01': ['*' if cvm_result.pvalue < 0.01 else '']
})

# Display the results
normality_results


# In[320]:


# Parameters for the simulation
min_sample_size = 20     # Minimum sample size
max_sample_size = 1000   # Maximum sample size
step_size = 10           # Step size for increasing sample size
num_simulations = 100     # Number of simulations per sample size
dot_size = 10            # Size of the dots in the plot

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
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Shapiro-Wilk p-value'], alpha=0.5, s=dot_size, c="blue")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Shapiro-Wilk p-value')
plt.title('Sample Size vs. Average Shapiro-Wilk p-value (across simulations)')
plt.legend()
plt.show()


# In[321]:


# Plot residuals to check for homoscedasticity
def plot_residuals(model):
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Scatter plot of residuals vs fitted values
    plt.scatter(fitted_values, residuals, alpha=0.1, s=5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.show()

# Example usage:
plot_residuals(model)


# In[322]:


# Extract residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Parameters for the simulation
min_sample_size = 20      # Minimum sample size
max_sample_size = 6000    # Maximum sample size
step_size = 100            # Step size for increasing sample size
num_simulations = 10     # Number of simulations per sample size
dot_size = 10             # Size of the dots in the plot

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

# Plot the Sample Size vs. Average Breusch-Pagan P-Value curve
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Sample Size'], results_df['Average Breusch-Pagan p-value'], alpha=1, s=dot_size, c="purple")
plt.axhline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
plt.axhline(0.10, color='orange', linestyle='--', label='p=0.10 threshold')
plt.xlabel('Sample Size')
plt.ylabel('Average Breusch-Pagan p-value')
plt.title('Sample Size vs. Average Breusch-Pagan p-value (across simulations)')
plt.legend()
plt.show()


# # Partial ETA-squared for individual surveys

# ### Ai-jobs.net

# In[325]:


# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'country', 'company_size']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_ai_w_l[var] = df_ai_w_l[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_ai_w_l).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[326]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_ai_w_l.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_combined.groupby(var_name, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_combined.groupby(var_name, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# ### Kaggle

# In[328]:


df_k_w_l.head(1)


# In[329]:


# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'country','education_level','company_size','industry']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_k_w_l[var] = df_k_w_l[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_k_w_l).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[330]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_k_w_l.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_combined.groupby(var_name, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_combined.groupby(var_name,observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# ### DE-IT

# In[332]:


df_it_w_l.head(1)


# In[333]:


# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'company_industry', 'language_category', 'company_size','city']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_it_w_l[var] = df_it_w_l[var].astype('category')

# Define the formula for the model
formula = 'salary_2024_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_it_w_l).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[334]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_it_w_l.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_combined.groupby(var_name, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_combined.groupby(var_name, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# # Post-hoc power Analysis

# Use post-hoc power analysis to understand whether non-significant results are due to a lack of power or a true absence of effect.

# In[337]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.power import FTestAnovaPower


# In[338]:


# List of your categorical variables
categorical_vars = ['survey', 'seniority_level', 'job_category', 'year', 'country']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_combined[var] = df_combined[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_combined).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[339]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_combined.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# In[340]:


# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'company_industry', 'language_category', 'company_size']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_it_w_l[var] = df_it_w_l[var].astype('category')

# Define the formula for the model
formula = 'salary_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_it_w_l).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Display the results
anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq']]


# In[341]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_it_w_l.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# Statistical Power: The probability that a test will correctly reject a false null hypothesis (i.e., detect a true effect when it exists). A commonly accepted threshold for adequate power is 0.80 (or 80%).\
# We can be more confident that there is no significant effect associated with this variable in the data.\
# The study was adequately powered to detect a meaningful effect if one existed.\
# \
# Variables with Significant P-value and Adequate Power: Confirm the effect and consider its practical implications.\
# Variables with High Power and High P-value: Conclude that there is likely no significant effect. Focus resources on other variables with potential effects.

# In[343]:


# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Function to extract variable name from 'C(variable_name)'
def extract_variable_name(var):
    if var.startswith('C(') and var.endswith(')'):
        return var[2:-1]
    else:
        return var

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_it_w_l.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var in variables:
    # Get the actual variable name without 'C()'
    var_name = extract_variable_name(var)
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the variable
    k_groups = df_combined[var_name].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_combined.groupby(var_name, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_combined.groupby(var_name, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])


# ## Try: 17:20

# In[345]:


# Define the dataframe to be used
df_name = df_ai_w_l  # You can update df_combined to any other dataframe as needed

# List of your categorical variables
categorical_vars = ['seniority_level', 'job_category', 'year', 'country', 'company_size']

# Convert variables to 'category' data type
for var in categorical_vars:
    df_name[var] = df_name[var].astype('category')

# Define the formula for the model
formula = 'salary_norm_log ~ ' + ' + '.join(['C(' + var + ')' for var in categorical_vars])

# Fit the model
model = ols(formula, data=df_name).fit()

# Perform ANOVA with Type III sum of squares
anova_table = sm.stats.anova_lm(model, typ=3)

# Calculate total sum of squares excluding 'Residual' and 'Intercept'
ss_total = anova_table.loc[anova_table.index != 'Residual', 'sum_sq'].sum()

# Get the sum of squares for residual
ss_residual = anova_table.loc['Residual', 'sum_sq']

# Variables to calculate eta-squared for
variables = [var for var in anova_table.index if var not in ['Residual', 'Intercept']]

# Initialize dictionaries to store eta-squared values
eta_sq = {}
partial_eta_sq = {}

for var in variables:
    ss_effect = anova_table.loc[var, 'sum_sq']
    eta_sq[var] = ss_effect / ss_total
    partial_eta_sq[var] = ss_effect / (ss_effect + ss_residual)

# Create DataFrame for eta-squared values
eta_sq_df = pd.DataFrame({
    'Eta_sq': eta_sq,
    'Partial_eta_sq': partial_eta_sq
})

# Merge with the ANOVA table
anova_results = anova_table.loc[variables].join(eta_sq_df)

# Function to calculate Cohen's f from partial eta-squared
def cohen_f_from_partial_eta_squared(eta_squared):
    if eta_squared >= 1 or eta_squared < 0:
        raise ValueError(f"Invalid eta-squared value: {eta_squared}. It must be between 0 and 1.")
    return np.sqrt(eta_squared / (1 - eta_squared))

# Initialize a dictionary to store power values
power_results = {}

# Total sample size
nobs = df_name.shape[0]

# Set alpha level (commonly 0.05)
alpha = 0.05

for var, actual_var in zip(variables, categorical_vars):
    # Get partial eta squared
    partial_eta_sq_var = anova_results.loc[var, 'Partial_eta_sq']
    # Compute Cohen's f
    effect_size = cohen_f_from_partial_eta_squared(partial_eta_sq_var)
    # Number of groups (levels) in the actual variable
    k_groups = df_name[actual_var].nunique()
    # Calculate the effective sample size for this variable
    # Sum of observations across levels of the factor
    nobs_var = df_name.groupby(actual_var, observed=True).size().sum()
    # Alternatively, use the harmonic mean of group sizes
    group_sizes = df_name.groupby(actual_var, observed=True).size().values
    nobs_var = len(group_sizes) * (len(group_sizes) / np.sum(1 / group_sizes))
    # Compute power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size=effect_size, nobs=nobs_var, alpha=alpha, k_groups=k_groups)
    # Store the power value
    power_results[var] = power

# Add power values to the anova_results DataFrame
anova_results['Power'] = pd.Series(power_results)

# Display the updated ANOVA results with power
display(anova_results[['sum_sq', 'df', 'F', 'PR(>F)', 'Eta_sq', 'Partial_eta_sq', 'Power']])

