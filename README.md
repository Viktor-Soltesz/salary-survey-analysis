# salary-survey-analysis

## Statistical Analysis (Stage 6 of 6)

This repository performs inferential and predictive analysis using the cleaned and modeled salary survey data from earlier pipeline stages.  
It uses Jupyter notebooks, `pandas`, `statsmodels`, and `scipy` to run robust statistical tests (e.g. ANOVA) and multiregression models to uncover trends and drivers of developer compensation across countries, roles, and seniority levels.

---

## Project Overview

This project is split into modular repositories, each handling one part of the full ELT and analytics pipeline:

| Stage | Name                        | Description                                | Repository |
|-------|-----------------------------|--------------------------------------------|------------|
| 1     | Ingestion & Infrastructure  | Terraform + Cloud Functions for ETL        | [salary-survey-iac](https://github.com/Viktor-Soltesz/salary-survey-iac) |
| 2     | Modeling & Transformation   | DBT models, metrics, testing               | [salary-survey-dbt](https://github.com/Viktor-Soltesz/salary-survey-dbt) |
| 3     | Tableau dashboards          | Interactive salary exploration             | [Tableau Public](https://public.tableau.com/app/profile/viktor.solt.sz/viz/SoftwareDeveloperSalaries/Dashboard) |
| 4     | Model Observability         | Drift & lineage tracking (Elementary)      | [salary-survey-edr](https://github.com/Viktor-Soltesz/salary-survey-edr) |
| 5     | Data Quality Monitoring     | Great Expectations (GX) data observability  | [salary-survey-gx](https://github.com/Viktor-Soltesz/salary-survey-gx) |
| **▶️ 6** | **Statistical Analysis**    | **ANOVA, regressions, prediction**   | **[salary-survey-analysis](https://github.com/Viktor-Soltesz/salary-survey-analysis)** |
| +     | Presentation   |  [Google slides](https://docs.google.com/presentation/d/1BHC6QnSpObVpulEcyDLXkW-6YLo2hpnwQ3miQg43iBg/edit?slide=id.g3353e8463a7_0_28#slide=id.g3353e8463a7_0_28) |
| +     | Data Dictionary | [Google sheets](https://docs.google.com/spreadsheets/d/1cTikHNzcw3e-gH3N8F4VX-viYlCeLbm5JkFE3Wdcnjo/edit?gid=0#gid=0) |

---

## Repository Scope

This repository focuses on:
- **Exploratory data analysis** (EDA) of cleaned survey data
- **Multivariate ANOVA** to test how various factors affect salary
- **Regression modeling** to predict and quantify salary based on profile features
- **Post-hoc tests, diagnostics, and assumption validation**
- **Insights communication** through annotated notebooks

It complements the BI dashboard by offering controlled, reproducible quantitative conclusions beyond visual trends.

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

---

## Appendix

- Reproducible via Jupyter Lab
- Libraries used: `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`
- Notebooks include inline commentary and visualizations
- Supports interpretation of results shown in Tableau (Stage 3)

---

