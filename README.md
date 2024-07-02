# Industry-Trend-Analysis

# Industry Trend Analysis Project

This project focuses on analyzing a dataset of organizations to uncover trends and insights through data wrangling, data mining techniques, and data visualization.

## Table of Contents
- [Introduction](#introduction)
- [Data Wrangling](#data-wrangling)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Mining Techniques](#data-mining-techniques)
  - [Cluster Analysis](#cluster-analysis)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Text Mining](#text-mining)
- [Data Visualization](#data-visualization)
- [Results Summary](#results-summary)
- [Dependencies](#dependencies)

## Introduction
This project aims to perform comprehensive analysis on a dataset of 10,000 organizations to understand various industry trends, perform clustering, classification, regression, and text mining, and visualize the results.

## Data Wrangling
- **Loading and Cleaning**: Loaded the dataset from a CSV file and performed data cleaning tasks including handling missing values, removing duplicates, and standardizing text fields.
- **Data Summary**: Provided a summary of the dataset's structure and transformations.

## Exploratory Data Analysis (EDA)
- **Numerical Summary**: Described numerical columns.
- **Value Counts**: Displayed counts of 'Industry' and 'Country' fields.
- **Correlation Matrix**: Analyzed correlations among numerical features.

## Data Mining Techniques
### Cluster Analysis
- Used KMeans clustering on standardized 'Founded' and 'Number of employees' fields.
- Visualized clusters using PCA.

### Classification
- Applied Naive Bayes classifier to predict 'Industry' for top 5 industries.
- Provided classification report and confusion matrix.

### Regression
- Performed linear regression to predict 'Number of employees' based on 'Founded' year.
- Displayed regression coefficients and intercept.

### Text Mining
- Utilized TF-IDF for extracting top terms from organization descriptions.

## Data Visualization
- **Bar Chart**: Top 10 industries by number of organizations.
- **Pie Chart**: Top 5 countries by number of organizations.
- **Boxplot**: Distribution of number of employees by industry.
- **Line Chart**: Average number of employees over years.
- **Multi-Line Chart**: Employee trends in top 5 industries.
- **Scatter Plot**: Relationship between founded year and number of employees.

## Results Summary
- Cluster analysis identified distinct groups within the dataset.
- Classification achieved reasonable accuracy for predicting top industries.
- Regression showed trends in employee numbers based on founded year.
- Text mining highlighted significant terms from organization descriptions.

## Dependencies
- Python 3.x
- NumPy
- pandas
- seaborn
- scikit-learn
- matplotlib

## How to Run
1. Ensure all dependencies are installed.
2. Load the dataset `organizations-10000.csv` in the same directory as the script.
3. Run the script to perform data analysis and generate visualizations.

## License
This project is licensed under the MIT License.

