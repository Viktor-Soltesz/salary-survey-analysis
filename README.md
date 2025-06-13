# Statistical Modeling (Stage 4 of 5)

This repository performs inferential and predictive analysis using the cleaned and modeled salary survey data from earlier pipeline stages.  
It uses Jupyter notebooks, `pandas`, `statsmodels`, and `scipy` to run robust statistical inference (ANOVA) and multiregression models to quantify the drivers of Salaries of Software Developers across countries, roles, seniority levels, and other factors.

---

## Project Overview

This project is split into modular repositories, each handling one part of the full ELT and analytics pipeline:

| Stage | Name                        | Description                                | Link |
|-------|-----------------------------|--------------------------------------------|------------|
| ㅤ1     | Ingestion & Infrastructure  | Terraform + Python Cloud Functions        | [salary-survey-iac (GitHub)](https://github.com/Viktor-Soltesz/salary-survey-iac) |
| ㅤ2     | Data Transformation   | DBT data models and testing               | [salary-survey-dbt (GitHub)](https://github.com/Viktor-Soltesz/salary-survey-dbt) <br> ㅤ⤷ [DBT docs](https://viktor-soltesz.github.io/salary-survey-dbt-docs/index.html#!/overview)|
| ㅤ3     | Data Observability  | Great Expectations & Elementary, <br> model monitoring and data observability     | [salary-survey-gx (GitHub)](https://github.com/Viktor-Soltesz/salary-survey-gx) <br> ㅤ⤷ [GX log](https://viktor-soltesz.github.io/salary-survey-gx/gx_site/index.html) <br> ㅤ⤷ [Elementary report](https://viktor-soltesz.github.io/salary-survey-dbt/elementary_report.html#/report/dashboard) |
| **▶️4** | **Statistical Modeling**    | **ANOVA, multiregressions, prediction**   | **[salary-survey-analysis (GitHub)](https://github.com/Viktor-Soltesz/salary-survey-analysis)** |
| ㅤ5     | Dashboards          | •ㅤInteractive salary exploration <br> •ㅤData Health metrics (from DBT) <br> •ㅤBilling report (from GCP invoicing) <br> •ㅤBigQuery report (from GCP logging) |ㅤ🡢 [Tableau Public](https://public.tableau.com/app/profile/viktor.solt.sz/viz/SoftwareDeveloperSalaries/Dashboard) <br>ㅤ🡢 [Looker Studio](https://lookerstudio.google.com/s/mhwL6JfNlaw)<br>ㅤ🡢 [Looker Studio](https://lookerstudio.google.com/s/tp8jUo4oPRs)<br>ㅤ🡢 [Looker Studio](https://lookerstudio.google.com/s/v2BIFW-_Jak)|
| ㅤ+     | Extra material | •ㅤPresentation <br> •ㅤData Dictionary <br>  •ㅤSLA Table <br>  •ㅤMy LinkedIn<br>  •ㅤMy CV|ㅤ🡢 [Google Slides](https://docs.google.com/presentation/d/1BHC6QnSpObVpulEcyDLXkW-6YLo2hpnwQ3miQg43iBg/edit?slide=id.g3353e8463a7_0_28#slide=id.g3353e8463a7_0_28) <br>ㅤ🡢 [Google Sheets](https://docs.google.com/spreadsheets/d/1cTikHNzcw3e-gH3N8F4VX-viYlCeLbm5JkFE3Wdcnjo/edit?gid=0#gid=0) <br>ㅤ🡢 [Google Sheets](https://docs.google.com/spreadsheets/d/1r85NlwsGV1DDy4eRBfMjZgI-1_uyIbl1fUazgY00Kz0/edit?usp=sharing) <br>ㅤ🡢 [LinkedIn](https://www.linkedin.com/in/viktor-soltesz/) <br>ㅤ🡢 [Google Docs](https://www.linkedin.com/in/viktor-soltesz/)|

---

## Repository Scope

This repository focuses on:
- **Exploratory data analysis** (EDA) of cleaned survey data
- **Multivariate ANOVA** to test how various factors affect salary
- **Regression modeling** to predict and quantify salary based on profile features
- **Post-hoc tests, diagnostics, and assumption validation**

It complements the Tableau dashboard by offering controlled, reproducible quantitative conclusions beyond visual trends.

---

## Detailed Breakdown

### 1. Project Overview:
- This project analyzes software developer salaries using thousands of survey responses collected over multiple years (2018-2024).

---

### 2. Objectives:
The project aims to:
- Understand the software developer field.
- Examine and evaluate key variables influencing salaries; their variability and significance.
- Determine if all the surveys consistently reflect the same underlying salary trends.
- Develop a model to predict a personalized expected salary.

---

### 3. Data Sources:
- The analysis builds on 11 publicly available surveys from 3 distinct sources, covering a range of data points on developer salaries.
  - Kaggle: https://www.kaggle.com/code/holfyuen/kaggle-survey-2017-2022/input
  - AI-Jobs.net:	https://ai-jobs.net/salaries/
  - Yearly Germany: IT Salary Survey https://www.asdcode.de/2024/04/it-salary-survey-december-2023.html

---

### 4. Project Phases:

- 4.1. Preparation
  - Standardizes the survey data through data cleaning, transformation, and consolidation into a common format.
  - Addresses missing data, identifies and removes outliers, and reports data quality metrics.

- 4.2. Analysis
  - Explores respondent profiles and identifies trends in the data.
  - Conducts a multivariate ANOVA to compare surveys and quantify the impact of various variables on salary.

- 4.3. Prediction
  - Begins with a baseline multilinear regression model using key variables that are verifiable with Glassdoor data.
  - Refines the model by adding additional parameters not covered by Glassdoor, such as language proficiency, industry, and company size, to create a personalized salary estimate.
  - And at the end, the coefficients of a multiregression model will yield the effect of the independent parameters which are normalized properly with all the other variables. 

### 5. Usage
This project serves as a guide for those interested in exploring salary trends 
in the software development industry, providing insights into data preparation, 
analysis, and predictive modeling for salary estimation.

- Reproducible via Jupyter Notebooks, but for best performance I recommend Jupyter Lab.
- Create a new virtual environment and install the dependencies in `requirements.txt`
- Happy Exploration :)
---

