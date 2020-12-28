# Dementia

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Table of Content
================

- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Codebooks](#codebooks)
- [Imputation and Scaling](#imputation-and-scaling)
- [Colro Code](#color-code)
- [Run the code properly](#run-the-code-properly)
- [Advanced ML Evaluation Metrics](#advanced-ml-evaluation-metrics)
- [Requirements](#requirements)

This is the repository for the dementia project

# Repository Structure

  - **Input Folder**: Folder Containing main input datasets used for processing
  - **Code Folder**: Folder containing main codes
  - **Distribution Folder**: Folder containing distribution plots of all features
  - **visualization_tools Folder**: Folder for Google Facets \& what-if tool

# Datasets:

  - [Copy of Dementia_baseline_questionnaire_V1.xlsx](https://bitbucket.org/HiyamGh/dementia/src/master/input/Copy%20of%20Dementia_baseline_questionnaire_V1.xlsx): Beqaa Questionnaire
  - [pooled_data.xlsx](https://bitbucket.org/HiyamGh/dementia/src/master/input/pooled_data.xlsx): pooled data - xlsx format
  - [pooled_data.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/pooled_data.csv): pooled data - csv format
  - [pooled_new.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/pooled_new.csv): pooled data after replacing the erroneous values in ***ANIMALS_2*** with 1s

# Codebooks

  - [numeric.xlsx](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/numeric.xlsx): numeric features and their statistics
  - [textual.xlsx](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/textual.xlsx): textual features and their statistics
  - [missing_40_codebook.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/missing_40_codebook.csv): Code-book for the features having **greater than** 40% missing values.
  - [missing_codebook_lessthan_40.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/missing_codebook_lessthan_40.csv) Code-book for the features having **less than** 40% missing values.
  - [num_rows_missing_codebook.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/num_rows_missing_entries_codebook.csv): Code-book for the number of rows with $i$ missing entries, as indicated below:
  - [erroneous_codebook.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/erroneous_codebook.csv): Code-book for the **erroneous values**, their **percentage**, and the **cut-off** based on which we defined what values are erroneous, for each feature
  - [jump_informant_columns.csv](https://bitbucket.org/HiyamGh/dementia/src/master/input/codebooks/jump_informant_columns.csv): Code-book for all columns that require jumps to other columns along with the informant ones. The columns are **sorted by increasing order of missing values**. For each column we indicate whether it is **INFORMANT** or not. A description is also included for these columns.
  >  Please write a loop that goes over all the rows, and produces the following: for $i= 1$ to $n$ where $n$ is the total number of columns, produce x_i, the number of rows with $i$ missing entries.

# Imputation and Scaling

  - Imputed missing values of all **numeric/ordinal** features using KNN
  - Imputed missing values of all **categorical** features by replacing all **nan**s with a new **category**.


# Color-Code

  The **yellow color** stands for **carer**, the **red color** stands for **patient**

# Advanced ML Evaluation Metrics
  - Quantifying 'risk' as being the probability of the positive class
  - Mean Empirical Risk Curves
  - Precision/Recall Curves at Top K
  - Probability of Mistake per model per **Frequent Pattern** Using FP-growth technique
    * Use percentiles(distribution) to "itemize"/"categorize" values inside column
  - Jaccard Similarity curves between model pairs
  - All models used here are shallow models - Could incorporate Deep Learning Models soon.
  - SMOTE used to oversample **training data** before testing (DO NOT APPLY SMOTE ON THE WHOLE DATA THEN DO A TRAIN-TEST SPLIY)


# Run the Code properly
```
cd Code

# creating meta data first
python create_features_meta.py
python create_features_with_categories.py

# then, creating numeric & textual
python create_numeric.py
python create_textual.py

# produce code-books
python create_codebooks.py
```
# Requirements
   - XlsxWriter ```1.2.8```
   - numpy ```1.18.5```
   - tensorflow ```1.12.0```
   - seaborn ```0.9.0```
   - xgboost ```1.0.2```
   - matplotlib ```3.0.3```
   - scipy ```1.4.1```
   - pandas ```0.24.0```
   - scikit_learn ```0.23.2```
