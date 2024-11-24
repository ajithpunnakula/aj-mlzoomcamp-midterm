# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import joblib
import pickle

# Load the dataset
df_train = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')

# Clean column names by removing spaces and converting to lowercase
df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
print(f"Number of input columns: {len(df_train.columns)}")

# Initialize a list to keep track of all dropped columns
dropped_columns = []

# Drop columns that are not useful for the model
columns_to_drop = ['id']
df_train = df_train.drop(columns=columns_to_drop, errors='ignore')
dropped_columns.extend(columns_to_drop)

# Drop columns with a single value
single_value_columns = df_train.columns[df_train.nunique() <= 1].tolist()
print(f"Columns with a single value or all the same: {single_value_columns}")
df_train = df_train.drop(columns=single_value_columns, errors='ignore')
dropped_columns.extend(single_value_columns)

# Drop columns with a high percentage of missing values
high_missing_cols = df_train.columns[df_train.isnull().mean() > 0.5].tolist()
print(f"Columns with a high percentage of missing values: {high_missing_cols}")
df_train = df_train.drop(columns=high_missing_cols, errors='ignore')
dropped_columns.extend(high_missing_cols)

# Drop columns with a high percentage of zero values
high_zero_cols = df_train.columns[(df_train == 0).mean() > 0.5].tolist()
print(f"Columns with a high percentage of zero values: {high_zero_cols}")
df_train = df_train.drop(columns=high_zero_cols, errors='ignore')
dropped_columns.extend(high_zero_cols)

# Numerical columns and categorical columns
numerical_columns = df_train.select_dtypes(include=['number']).columns
categorical_columns = df_train.select_dtypes(exclude=['number']).columns

# Fill missing values in numerical columns with the mean
df_train[numerical_columns] = df_train[numerical_columns].fillna(df_train[numerical_columns].mean())
# Fill missing values in categorical columns with the mode
df_train[categorical_columns] = df_train[categorical_columns].fillna(df_train[categorical_columns].mode().iloc[0])

# Drop high and low correlated columns
correlation_matrix = df_train[numerical_columns].corr().abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
low_corr_cols = [column for column in upper_tri.columns if all(upper_tri[column] < 0.1)]
correlation_based_drop = list(set(high_corr_cols + low_corr_cols))
print(f"Columns to drop due to correlation: {correlation_based_drop}")
df_train = df_train.drop(columns=correlation_based_drop, errors='ignore')
dropped_columns.extend(correlation_based_drop)
# Print all dropped columns
print(f"All dropped columns: {dropped_columns}")
with open('./outputs/dropped_columns.pkl', 'wb') as f:  
    pickle.dump(dropped_columns, f)

# Perform one-hot encoding on categorical columns
dv = DictVectorizer(sparse=False)
df_encoded = dv.fit_transform(df_train[categorical_columns].to_dict(orient='records'))
df_encoded = pd.DataFrame(df_encoded, columns=dv.get_feature_names_out())
df_train = pd.concat([df_train.drop(columns=categorical_columns, errors='ignore'), df_encoded], axis=1)
df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
# Save DictVectorizer for later use
joblib.dump(dv, './outputs/dv.joblib')

# Plot the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(df_train['saleprice'], kde=True)
plt.title('Distribution of Sale Price')
plt.show()

# Plot the distribution of the target variable after log transformation
plt.figure(figsize=(12, 6))
sns.histplot(np.log1p(df_train['saleprice']), kde=True)
plt.title('Distribution of Sale Price after Log Transformation')
plt.show()

# Apply log transformation to the target variable
df_train['saleprice'] = np.log1p(df_train['saleprice'])

# save the cleaned dataset
df_train.to_csv('./outputs/cleaned_training_dataset.csv', index=False)

# %%
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# load the cleaned dataset
df_train = pd.read_csv('./outputs/cleaned_training_dataset.csv')
# Assume df_* is your DataFrame with 'saleprice' as the target variable
X = df_train.drop(columns=['saleprice'])
Y = df_train['saleprice']

# Split the dataset into training and validation datasets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

def print_best_params_and_score(model_name, grid_search):
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV Mean Squared Error for {model_name}: {grid_search.best_score_}")

# Function to evaluate and store results
def evaluate_model(model_name, model, params=None):
    if params:
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train, Y_train)
        best_model = grid
        print_best_params_and_score(model_name, grid)
    else:
        model.fit(X_train, Y_train)
        best_model = model

    # Evaluate on validation data
    Y_pred = best_model.predict(X_val)
    mse = mean_squared_error(Y_val, Y_pred)
    print(f"Validation Mean Squared Error ({model_name}): {mse}")
    return model_name, mse, best_model

# Dictionary to hold results
model_performance = {}

# Linear Regression
name, mse, model = evaluate_model("Linear Regression", LinearRegression())
model_performance[name] = (mse, model)

# Lasso Regression
params_lasso = {'alpha': [0.01, 0.1, 1, 10]}
name, mse, model = evaluate_model("Lasso Regression", Lasso(), params_lasso)
model_performance[name] = (mse, model)

# Ridge Regression
params_ridge = {'alpha': [0.01, 0.1, 1, 10]}
name, mse, model = evaluate_model("Ridge Regression", Ridge(), params_ridge)
model_performance[name] = (mse, model)

# Random Forest Regression
params_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
name, mse, model = evaluate_model("Random Forest", RandomForestRegressor(random_state=42), params_rf)
model_performance[name] = (mse, model)

# Gradient Boosting Regression
params_gb = {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1, 0.2]}
name, mse, model = evaluate_model("Gradient Boosting", GradientBoostingRegressor(random_state=42), params_gb)
model_performance[name] = (mse, model)

# XGBoost Regression
params_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1, 0.2]}
name, mse, model = evaluate_model("XGBoost", XGBRegressor(random_state=42, objective='reg:squarederror'), params_xgb)
model_performance[name] = (mse, model)

# LightGBM Regression
params_lgbm = {'n_estimators': [100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1, 0.2], 'verbosity': [-1]}
name, mse, model = evaluate_model("LightGBM", LGBMRegressor(random_state=42), params_lgbm)
model_performance[name] = (mse, model)

# Identify the best model based on lowest MSE
best_model_name, (best_mse, best_model) = min(model_performance.items(), key=lambda x: x[1][0])
print(f"\nThe best model is {best_model_name} with a validation MSE of {best_mse}")

# Save the best model using pickle
with open(f"./outputs/best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
print(f"The best model {best_model_name} has been saved as 'best_model.pkl'")



# %%
import pandas as pd  
import numpy as np  
import pickle  
import joblib  
from sklearn.feature_extraction import DictVectorizer
# Load the test dataset
df_test = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')

# Clean column names similarly to how it's done in df_train
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()

dropped_columns = None
with open('./outputs/dropped_columns.pkl', 'rb') as f:  
    dropped_columns = pickle.load(f)  
# Drop columns as done in the training data preprocessing
df_test = df_test.drop(columns=dropped_columns, errors='ignore')

# Separate numerical and categorical columns
numerical_columns_test = df_test.select_dtypes(include=['number']).columns
categorical_columns_test = df_test.select_dtypes(exclude=['number']).columns

# Fill missing values in test set using the same statistics from training set
df_test[numerical_columns_test] = df_test[numerical_columns_test].fillna(df_test[numerical_columns_test].mean())
df_test[categorical_columns_test] = df_test[categorical_columns_test].fillna(df_test[categorical_columns_test].mode().iloc[0])

# # Load the DictVectorizer fitted earlier
dv = joblib.load('./outputs/dv.joblib')

# Apply the DictVectorizer to encode categorical features in the test dataset
df_test_encoded = dv.transform(df_test[categorical_columns_test].to_dict(orient='records'))
df_test_encoded = pd.DataFrame(df_test_encoded, columns=dv.get_feature_names_out())

# Combine one-hot encoded features with non-categorical features
df_test = pd.concat([df_test.drop(columns=categorical_columns_test, errors='ignore'), df_test_encoded], axis=1)
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()
X_test = df_test

# Split the dataset into training and validation datasets
best_model = None
# load the best model
with open("./outputs/best_model.pkl", "rb") as model_file:
    best_model = pickle.load(model_file)

# Evaluate the best model on the test data
Y_test_pred_log = best_model.predict(X_test)
# Reverse transform to get predictions in the original target scale  
Y_test_pred_original = np.expm1(Y_test_pred_log)
print(Y_test_pred_original)


