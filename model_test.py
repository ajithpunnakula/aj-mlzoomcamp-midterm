# %%
import pandas as pd  
import numpy as np  
import pickle  
import joblib  
from sklearn.feature_extraction import DictVectorizer
# Load the test dataset
df_test = pd.read_csv('./inputs/test.csv')

# Clean column names similarly to how it's done in df_train
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()

dropped_columns = None
with open('./outputs/dropped_columns.pkl', 'rb') as f:  
    dropped_columns = pickle.load(f)  
# Drop columns as done in the training data preprocessing
df_test = df_test.drop(columns=dropped_columns, errors='ignore')

# Load the DictVectorizer and model  
dv = joblib.load('./outputs/dv.joblib')  
with open("./outputs/best_model.pkl", "rb") as model_file:  
    best_model = pickle.load(model_file)  

# Fill missing values in test set using the same statistics from training set
for col in df_test.select_dtypes(include=[np.number]):  
    df_test[col] = df_test[col].fillna(df_test[col].mean())  
  
for col in df_test.select_dtypes(exclude=[np.number]):  
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0]) 

# Apply the DictVectorizer to encode categorical features in the test dataset  
df_test_encoded = dv.transform(df_test.to_dict(orient='records'))  
X_test = pd.DataFrame(df_test_encoded, columns=dv.get_feature_names_out())  

# convert space to underscore and lowercase
X_test.columns = X_test.columns.str.replace(' ', '_').str.lower()
# Ensure order and completeness by aligning with the feature set that the model expects  
expected_features = list(best_model.feature_names_in_) 
X_test = X_test.reindex(columns=expected_features, fill_value=0)  
# Evaluate the model on the test data  
Y_test_pred_log = best_model.predict(X_test)  
# Reverse transform to get predictions in the original target scale  
Y_test_pred_original = np.expm1(Y_test_pred_log)  
print(Y_test_pred_original)  