# What Drives the Price of a Car?

## Assignment Notebook
[Click here to view the assignment notebook](https://github.com/ojbskvasu/ML-Assignment2/blob/main/prompt_II.ipynb)

## Problem Statement
understand what factors make a used car more or less expensive and to provide clear recommendations to a used car dealership as to what consumers value in a used car based on the analysis of the provided dataset.

## Data Overview
- **File Location:** `data/vehicles.csv`
- **Description:** Each row in the dataset represents an instance where a car.
- **Attributes:** For this analysis, a reduced dataset of approximately 426,000 used car listings was used to ensure efficient processing.

The dataset includes various attributes for each car listing, such as:

id: Unique identifier for each listing.
region: Geographical region of the listing.
price: The listed price of the car (target variable).
year: Manufacturing year.
manufacturer: Car brand.
model: Specific model.
condition: Reported condition.
cylinders: Number of cylinders.
fuel: Fuel type.
odometer: Mileage.
title_status: Title status.
transmission: Transmission type.
VIN: Vehicle identification number.
drive: Drivetrain.
size: Size category.
type: Body type.
paint_color: Exterior color.
state: State of listing.

## Key Findings
- **Data Analysis and Preparation:**
- key findings from the Data Analysis and Data Preparation steps:

Handling Missing Values: We identified columns with a significant number of missing values. Specifically, the 'size' column had over 70% missing values, so we decided to drop it entirely. For other columns with fewer missing values, we imputed them. For numerical columns like 'year' and 'odometer', we filled the missing values with the median of the respective columns. For categorical columns, we used the mode (the most frequent value) to fill in the missing entries.
Handling Outliers: During the initial data exploration, we noticed some extreme values in 'price', 'year', and 'odometer' that could disproportionately affect our models. We removed rows where the price was over $1,000,000, the manufacturing year was before 1980, or the odometer reading was over 1,000,000 miles. This helped to create a more representative dataset for typical used cars.
Data Type Conversion: We converted the data types of 'year' and 'odometer' from float64 to integer type since they represent whole numbers (years and miles).
Feature Engineering: We created two new features that we thought would be useful for predicting price:
car_age: Calculated by subtracting the car's manufacturing year from the current year. This directly represents how old the car is.
price_per_odometer: Calculated by dividing the car's price by its odometer reading (with a small value added to the odometer to avoid division by zero). This gives a measure of the price relative to the mileage.
Handling Categorical Features: Machine learning models typically require numerical input. We used one-hot encoding to convert the categorical columns (like 'manufacturer', 'fuel', 'transmission', etc.) into a numerical format. This creates new binary columns for each unique category. We also dropped the first dummy variable for each category to avoid multicollinearity, which can be an issue in some models.
Applying Transformations: To address skewed distributions in some numerical features ('odometer', 'car_age', and 'price_per_odometer'), we applied a logarithmic transformation (np.log1p). This can help models perform better by making the distributions more symmetrical.
Standard Scaling: We applied standard scaling to the other numerical features (excluding the target variable 'price' and the log-transformed features). Scaling standardizes the range of these features, which is important for many machine learning algorithms that are sensitive to the scale of input data.
Splitting Data: Finally, we split the prepared dataset into training and testing sets. The training set (80% of the data) is used to train our models, and the testing set (20% of the data) is used to evaluate their performance on unseen data.
These steps were crucial in preparing the raw data for building our regression models to predict used car prices..

- **Overall Acceptance Rate:** Approximately XX% of the coupons were accepted.


## Recommendations


## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ojbskvasu/ML-Assignment1.git
