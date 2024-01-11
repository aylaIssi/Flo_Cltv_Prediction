# Customer Lifetime Value (CLTV) Prediction
## Business Problem
The purpose of this project is to analyze customer behavior and develop effective marketing strategies for an online shoe store. The dataset provided contains information about customers' last purchases, and the goal is to segment customers based on their behavior and predict their lifetime value.

## Mission 1: Data Understanding and Preparation
Step 1: Import Libraries
The required libraries for data analysis, manipulation, and visualization are imported. This includes datetime, numpy, pandas, and matplotlib.

## Step 2: Read Dataset
The dataset is loaded from a CSV file into a pandas DataFrame named df.

## Step 3: Checking the Data
A function named datacheck is defined to perform basic data checks, such as displaying the first 10 rows, shape, info, describe, and checking for missing values.

## Step 4: Outlier Thresholds
Functions for handling outliers in numerical columns are defined. Outliers are replaced with corresponding threshold values.

## Step 5: Feature Engineering
New features are created by summing up total purchases and total customer expenses for omnichannel customers. Date columns are converted to datetime format.

## Mission 2: Creating CLTV Data Structure
### Step 1: Determine Observation Period
The maximum date from the "last_order_date" column is identified as the last date for analysis. Two datetime objects, last_date and today_date, are created.

### Step 2: CLTV DataFrame Creation
A DataFrame named cltv_df is created, containing customer ID, recency, T (tenure), frequency, and monetary value. Recency and T are calculated in weeks.

## Mission 3: Creating BG/NBD, Gamma Gamma Models, and Calculating CLTV
### Step 1: Fit the BG/NBD Model
The BetaGeoFitter is used to fit the BG/NBD model, displaying the model parameters (r, alpha, a, b).

### Step 2: Fit the Gamma Gamma Model
The GammaGammaFitter is used to fit the Gamma Gamma model, displaying the model parameters (p, q, v).

### Step 3: Calculate Expected Sales
Expected sales for 3 and 6 months are calculated using the BG/NBD model.

### Step 4: Calculate Probability Alive
The probability of a customer being alive is calculated using the BG/NBD model.

### Step 5: Calculate Expected Average Profit
Expected average profit is calculated using the Gamma Gamma model.

### Step 6: Calculate CLTV
Customer Lifetime Value (CLTV) is calculated using the BG/NBD and Gamma Gamma models for a specified time period (e.g., 6 or 12 months).

## Conclusion
The code provides a comprehensive analysis of customer behavior, segmentation, and lifetime value prediction using the BG/NBD and Gamma Gamma models. The created CLTV data structure includes key metrics for each customer, and the models help in making predictions for future sales and customer value. The results can be used for personalized marketing strategies and customer segmentation.
