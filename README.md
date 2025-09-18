# What Drives the Price of a Car?

## Assignment Notebook
[Click here to view the assignment notebook](https://github.com/ojbskvasu/ML-Assignment2/blob/main/prompt_II.ipynb)

## Problem Statement
Understand what factors make a used car more or less expensive and provide clear recommendations to a used car dealership as to what consumers value in a used car, based on the analysis of the provided data.

## Data Overview
- **File Location:** `data/vehicles.csv`
- **Description:** Each row in the dataset represents an instance where a car.
- **Attributes:** For this analysis, a reduced dataset of approximately 426,000 used car listings was used to ensure efficient processing.

  The dataset includes various attributes for each car listing, such as:

  - **id:** Unique identifier for each listing.
  - **region:** Geographical region of the listing.
  - **price:** The listed price of the car (target variable).
  - **year:** Manufacturing year.
  - **manufacturer:** Car brand.
  - **model:** Specific model.
  - **condition:** Reported condition.
  - **cylinders:** Number of cylinders.
  - **fuel:** Fuel type.
  - **odometer:** Mileage.
  - **title_status:** Title status.
  - **transmission:** Transmission type.
  - **VIN:** Vehicle identification number.
  - **drive:** Drivetrain.
  - **size:** Size category.
  - **type:** Body type.
  - **paint_color:** Exterior color.
  - **state:** State of listing.

## Key Findings
- **Data Analysis and Preparation:**
  - Key findings from the Data Analysis and Data Preparation steps:

    - **Handling Missing Values:**  
      We identified columns with a significant number of missing values. Specifically, the 'size' column had over 70% missing values, so we decided to drop it entirely. For other columns, appropriate imputation strategies were used.

    - **Handling Outliers:**  
      During the initial data exploration, we noticed some extreme values in 'price', 'year', and 'odometer' that could disproportionately affect our models. We removed rows where the price, year, or odometer fell outside reasonable boundaries.

    - **Data Type Conversion:**  
      We converted the data types of 'year' and 'odometer' from float64 to integer type since they represent whole numbers (years and miles).

    - **Feature Engineering:**  
      We created two new features that we thought would be useful for predicting price:
      - **car_age:** Calculated by subtracting the car's manufacturing year from the current year. This directly represents how old the car is.
      - **price_per_odometer:** Calculated by dividing the car's price by its odometer reading (with a small value added to the odometer to avoid division by zero). This gives a measure of the price relative to the car's usage.

    - **Handling Categorical Features:**  
      Machine learning models typically require numerical input. We used one-hot encoding to convert the categorical columns (like 'manufacturer', 'fuel', 'transmission', etc.) into binary columns.

    - **Applying Transformations:**  
      To address skewed distributions in some numerical features ('odometer', 'car_age', and 'price_per_odometer'), we applied a logarithmic transformation (`np.log1p`). This can help normalize the data and improve model performance.

    - **Standard Scaling:**  
      We applied standard scaling to the other numerical features (excluding the target variable 'price' and the log-transformed features). Scaling standardizes the range of these features.

    - **Splitting Data:**  
      Finally, we split the prepared dataset into training and testing sets. The training set (80% of the data) is used to train our models, and the testing set (20% of the data) is used to evaluate them.

  These steps were crucial in preparing the raw data for building our regression models to predict used car prices.

- **Modeling
The objective of the modeling phase was to build regression models that can predict the price of used cars based on the prepared dataset.

Models Used:

We trained and evaluated two common regression models:

Linear Regression: A basic linear model that predicts the target variable as a linear combination of the input features.
Lasso Regression: A linear model that incorporates L1 regularization, which helps in feature selection by shrinking the coefficients of less important features towards zero.
Evaluation:

The models were trained on the training dataset and evaluated on a separate testing dataset using the following metrics:

Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.
Mean Squared Error (MSE): Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
Root Mean Squared Error (RMSE): The square root of the MSE, providing an error metric in the same units as the target variable.
R-squared (R2): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
Cross-validation (with 5 folds) was also performed on the training data to obtain a more robust estimate of the models' performance and assess their generalization ability.

Results Summary:

Based on the evaluation metrics, both Linear Regression and Lasso Regression models provided reasonable performance in predicting car prices. Lasso Regression showed slightly better performance with a higher R-squared score on both the test set and during cross-validation, suggesting it might be slightly better at capturing the variance in car prices and potentially benefiting from the regularization in handling the features.

Further hyperparameter tuning and exploration of other regression models could potentially lead to improved performance.

## Recommendations

*To be added based on further analysis.*

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/ojbskvasu/ML-Assignment2.git
   ```
