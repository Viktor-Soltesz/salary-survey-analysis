##########################################################################################
##											##
##	Decoding Developer Salaries: A Multivariate Analysis and Predictive Model	##
##											##
##########################################################################################

1. Project Overview:
	This project analyzes software developer salaries using thousands of survey responses 
	collected over multiple years (2018-2024).

2. Objectives:
	The project aims to:
	- Understand the software development field.
	- Examine and evaluate key variables influencing salaries; their variability and significance.
	- Determine if all the surveys consistently reflect the same underlying salary trends.
	- Develop a model to predict a personalized expected salary.

3. Data Sources:
	The analysis builds on 11 publicly available surveys from 3 distinct sources, 
	covering a range of data points on developer salaries.

4. Project Phases:

	4.1. Preparation
		Standardizes the survey data through data cleaning, transformation, 
		and consolidation into a common format.
		Addresses missing data, identifies and removes outliers, and reports data quality metrics.

	4.2. Analysis
		Explores respondent profiles and identifies trends in the data.
		Conducts a multivariate ANOVA to compare surveys and 
		quantify the impact of various variables on salary.

	4.3. Prediction
		Begins with a baseline multilinear regression model using key variables 
		that are verifiable with Glassdoor data.
		Refines the model by adding additional parameters not covered by Glassdoor, 
		such as language proficiency, industry, and company size, 
		to create a personalized salary estimate.
		And at the end, the coefficients of a multiregression model will yield 
		the effect of the independent parameters which are normalized properly with all the other variables. 

5. Usage:
	This project serves as a guide for those interested in exploring salary trends 
	in the software development industry, providing insights into data preparation, 
	analysis, and predictive modeling for salary estimation.