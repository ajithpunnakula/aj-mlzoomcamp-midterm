# Table of Contents

1. [House Prices Prediction Project](#house-prices-prediction-project)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Environment Setup](#environment-setup)
5. [Usage](#usage)
   - [Predictions with Model Deployed on AWS Cloud](#predictions-with-model-deployed-on-aws-cloud)
   - [Training](#training)
   - [Predictions in Local Python Environment](#predictions-in-local-python-environment)
   - [Prediction on Local Docker](#prediction-on-local-docker)
6. [Project Structure](#project-structure)
   - [Key Files and Directories](#key-files-and-directories)
7. [Data Processing Summary](#data-processing-summary)
   - [Steps Involved](#steps-involved)
8. [Model Training Steps](#model-training-steps)
   - [Steps Involved](#steps-involved-1)
9. [API Endpoint for House Price Prediction](#api-endpoint-for-house-price-prediction)
   - [Overview](#overview)
   - [Key Components](#key-components)
   - [Prediction Process](#prediction-process)
   - [Error Handling](#error-handling)
   - [Example Request](#example-request)

# House Prices Prediction Project

This repository features a machine learning project designed to predict house prices using the Kaggle dataset ["House Prices: Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Project Overview

This project encompasses several key activities to develop and deploy a robust price prediction model:

1. **Data Cleansing**: Implemented various techniques, such as replacing nulls with mean/mode, one-hot encoding, to preprocess and cleanse the dataset, ensuring high data quality for model training.

2. **Model Training**: Trained multiple machine learning models to capture complex patterns and enhance predictive performance.

3. **Model Selection**: Evaluated models using Mean Squared Error (MSE) as the performance metric to identify the best-performing model.

4. **Dockerization**: Containerized the application using Docker, utilizing FastAPI to host the model within the Docker container for efficient deployment.

5. **Deployment**: Deployed the Docker image in local environments and on AWS EC2 instances, facilitating scalable and flexible access to the model.

6. **Prediction**: Performed predictions by interfacing with the deployed model both locally and on AWS, demonstrating the model's practical application in different deployment scenarios.


## Dataset

- **Source**: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Description**: The dataset contains numerous features about houses, including but not limited to area, number of rooms, year built, and neighborhood quality.

## Environment Setup

To set up the environment for this project, please follow these steps:

1. **Ensure Python 3.9+ is Installed**
   Make sure you have Python version 3.9 or higher installed on your system. You can verify your Python version by running:
   ```bash
   python --version
    ```

2. clone git repo
    ```bash
    git clone git@github.com:ajithpunnakula/aj-mlzoomcamp-midterm.git
    ```

3. Install requirements via pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### predictions with model deployed on AWS cloud 
```bash
python app_cloud_test.py 
```

### Training

```bash
python model_training.py
```

### predictions in local python environment
```bash
python model_test.py
```

### prediction on local docker
```bash
docker build -t midterm/aj-fastapi-app .
docker run -d -p 8000:8000 aj-fastapi-app
python appp_test.py
```

## Project Structure

### Key Files and Directories  
  
- **Dockerfile**: Defines the Docker container configuration for running the application.  
- **README.md**: This file documents the project's setup, structure, and usage.  
- **inputs**: Directory containing input data files for training and testing the model.  
- **outputs**: Directory containing model outputs, including performance metrics and trained model artifacts.  
- **requirements.txt**: Lists all Python dependencies required to run the application.  

- **model_training.py**: Script for training the regression model on the Kaggle dataset.  
- **model_test.py**: script to run predictions on kaggle test data set.
- **model_training_test.ipynb**: Jupyter notebook for exploratory data analysis and model evaluation during development.  

- **app.py**: The FastAPI application code that serves model predictions.  
- **app_test.py**: Script for testing the model deployment locally within Docker.  
- **app_cloud_test.py**: Script for testing the deployed model in a cloud environment.  

## Data Processing Summary

This section outlines the data processing steps applied to the training dataset in preparation for model training.

### Steps Involved

1. **Loading the Dataset**
   - The dataset is loaded from a CSV file using `pandas`.

2. **Column Name Formatting**
   - All column names are converted to lowercase and spaces are replaced with underscores for consistency.

3. **Dropping Unnecessary Columns**
   - Dropped columns that are IDs or not useful for model training.
   - Columns with a single unique value were removed.

4. **Handling Missing and Zero Values**
   - Columns with more than 50% missing values were dropped.
   - Columns with more than 50% zero values were also removed.

5. **Handling Numerical and Categorical Columns**
   - Missing values in numerical columns are filled with the mean of each column.
   - Missing values in categorical columns are filled with the mode (most frequent value) of each column.

6. **Correlation-Based Column Removal**
   - Columns with high (> 0.8) and low (< 0.1) correlations with other numerical columns were identified and removed.

7. **One-Hot Encoding**
   - Categorical variables were transformed using one-hot encoding to convert them into a numerical format suitable for model training.
   - The `DictVectorizer` was used and saved (`dv.joblib`) for future transformations.

8. **Target Variable Transformation**
   - The target variable `SalePrice` was visualized and then log-transformed to reduce skewness and improve model performance.
   - Plots of the `SalePrice` distribution before and after log transformation are generated to provide insights into the data's distribution.

9. **Saving the Processed Data**
   - The cleaned and transformed dataset is saved as `cleaned_training_dataset.csv`.
   - A list of all dropped columns is saved using `pickle` for reference.
    
All transformations and preprocessing steps are crucial for ensuring that the dataset is clean, balanced, and ready for effective machine learning model training.


## Model Training Steps

This section describes the model training process, including data preparation, model selection, evaluation, and storage.

### Steps Involved

1. **Load Cleaned Data**
   - The cleaned dataset is loaded from `cleaned_training_dataset.csv`.

2. **Data Splitting**
   - The dataset is divided into features (`X`) and the target variable (`saleprice`).
   - Data is split into training (80%) and validation (20%) sets using `train_test_split`.

3. **Model Evaluation Function**
   - Defined a function `evaluate_model` to perform grid search (if hyperparameters are provided), train models, evaluate their performance using Mean Squared Error (MSE) on validation data, and print best parameters and scores.

4. **Model Training and Optimization**
   - Multiple models were trained and tuned using `GridSearchCV` for hyperparameter optimization wherever applicable:
     - **Linear Regression**
     - **Lasso Regression** with hyperparameters: `alpha`
     - **Ridge Regression** with hyperparameters: `alpha`
     - **Random Forest Regressor** with hyperparameters: `n_estimators`, `max_depth`
     - **Gradient Boosting Regressor** with hyperparameters: `n_estimators`, `max_depth`, `learning_rate`
     - **XGBoost Regressor** with hyperparameters: `n_estimators`, `max_depth`, `learning_rate`
     - **LightGBM Regressor** with hyperparameters: `n_estimators`, `max_depth`, `learning_rate`

5. **Model Performance**
   - Calculated and printed the validation MSE for each model.
   - Stored each model's performance and hyperparameters.

6. **Best Model Selection**
   - Identified the best model based on the lowest validation MSE.

7. **Saving the Best Model**
   - The best-performing model was saved as `best_model.pkl` using `pickle` for reuse.

In summary, this rigorous process ensures the selection of the most effective model through systematic evaluation and comparison, optimizing its performance for house price prediction.

## API Endpoint for House Price Prediction

This section details the FastAPI-based application setup for predicting house prices using a pre-trained machine learning model.

### Overview

The application employs a trained machine learning model to estimate house prices based on provided features. It integrates a `DictVectorizer` for feature transformation and uses FastAPI to create an HTTP interface.

### Key Components

1. **Model and Vectorizer Loading**
   - The pre-trained model is loaded from `best_model.pkl`.
   - A `DictVectorizer` is loaded from `dv.joblib` to perform one-hot encoding on categorical features.
   - Information on dropped columns during data preprocessing is loaded from `dropped_columns.pkl`.

2. **Request Data Model**
   - A `HouseData` Pydantic model specifies the expected input features, including various numerical and categorical data pertinent to house properties.

3. **Prediction Endpoint**
   - `/predict`: A POST endpoint accepting JSON input conforming to the `HouseData` schema. It processes the input features to make predictions about house prices.

### Prediction Process

- Upon receiving data, spaces in feature names are replaced with underscores and converted to lowercase to ensure uniformity.
- Dropped columns are removed from the input to align with the model's expectations.
- One-hot encoding is applied to categorical features using the `DictVectorizer`.
- The processed input is aligned to match the expected features for prediction.
- The model predicts log-transformed house prices, which the application reverses to the original scale using `np.expm1`.
- Finally, the predicted price is returned as a JSON response.

### Error Handling

- If any error occurs during processing or prediction, an HTTP 500 error is raised, describing the issue.

### Example Request

To make a prediction, send a POST request to `/predict` with a JSON body containing all the required house features as specified in the `HouseData` model.

This structured architecture ensures reliable and accurate predictions, leveraging FastAPI to provide a fast and scalable web service.

