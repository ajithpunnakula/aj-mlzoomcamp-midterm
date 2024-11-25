from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

# Load the pre-trained DictVectorizer and model  
try:  
    dv = joblib.load('./outputs/dv.joblib')  
    with open("./outputs/best_model.pkl", "rb") as model_file:  
        model = pickle.load(model_file)  
except Exception as e:  
    raise RuntimeError("Error loading models or vectorizers: " + str(e))  

# Load dropped columns information  
dropped_columns = None  
with open('./outputs/dropped_columns.pkl', 'rb') as f:  
    dropped_columns = pickle.load(f)  

class HouseData(BaseModel):  
    Id: int  
    MSSubClass: int  
    MSZoning: str  
    LotFrontage: float
    LotArea: int  
    Street: str  
    Alley: str
    LotShape: str  
    LandContour: str  
    Utilities: str  
    LotConfig: str  
    LandSlope: str  
    Neighborhood: str  
    Condition1: str  
    Condition2: str  
    BldgType: str  
    HouseStyle: str  
    OverallQual: int  
    OverallCond: int  
    YearBuilt: int  
    YearRemodAdd: int  
    RoofStyle: str  
    RoofMatl: str  
    Exterior1st: str  
    Exterior2nd: str  
    MasVnrType: str
    MasVnrArea: float  
    ExterQual: str  
    ExterCond: str  
    Foundation: str  
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: int  
    BsmtFinType2: str
    BsmtFinSF2: int  
    BsmtUnfSF: int  
    TotalBsmtSF: int  
    Heating: str  
    HeatingQC: str  
    CentralAir: str  
    Electrical: str
    FirstFlrSF: int  
    SecondFlrSF: int  
    LowQualFinSF: int  
    GrLivArea: int  
    BsmtFullBath: int  
    BsmtHalfBath: int  
    FullBath: int  
    HalfBath: int  
    BedroomAbvGr: int  
    KitchenAbvGr: int  
    KitchenQual: str  
    TotRmsAbvGrd: int  
    Functional: str  
    Fireplaces: int  
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: float
    GarageFinish: str
    GarageCars: int  
    GarageArea: int  
    GarageQual: str
    GarageCond: str
    PavedDrive: str  
    WoodDeckSF: int  
    OpenPorchSF: int  
    EnclosedPorch: int  
    ThreeSsnPorch: int  # Renamed to adhere to Python identifier rules  
    ScreenPorch: int  
    PoolArea: int  
    PoolQC: str
    Fence: str
    MiscFeature: str
    MiscVal: int  
    MoSold: int  
    YrSold: int  
    SaleType: str  
    SaleCondition: str  

@app.post("/predict")
def predict(data: HouseData):
    try:
        # Transform the input data using dv
        input_dict = data.dict()
        # remove spaces and convert to lowercase
        input_dict = {k.replace(' ', '_').lower(): v for k, v in input_dict.items()}
        for col in dropped_columns:
            input_dict.pop(col, None) # Remove dropped columns from the input data

        input = dv.transform([input_dict])
        X_test = pd.DataFrame(input, columns=dv.get_feature_names_out()) 

        # Ensure order and completeness by aligning with the feature set that the model expects  
        expected_features = list(model.feature_names_in_) 
        X_test = X_test.reindex(columns=expected_features, fill_value=0)  

        # Make prediction
        prediction_log = model.predict(X_test)
        # Reverse transform to get predictions in the original target scale
        prediction = np.expm1(prediction_log)
        
        # Return the prediction
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    

