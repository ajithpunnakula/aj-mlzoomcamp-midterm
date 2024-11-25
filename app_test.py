import requests

import json

class TestFastAPI:

    @staticmethod
    def test_predict_endpoint():
        # Define the URL for the request
        url = "http://127.0.0.1:8000/predict"

        # Define the headers for the request
        headers = {
            "Content-Type": "application/json"
        }

        # Define the JSON payload
        data = {
            "Id": 1461,
            "MSSubClass": 20,
            "MSZoning": "RH",
            "LotFrontage": 80.0,
            "LotArea": 11622,
            "Street": "Pave",
            "Alley": "NA",
            "LotShape": "Reg",
            "LandContour": "Lvl",
            "Utilities": "AllPub",
            "LotConfig": "Inside",
            "LandSlope": "Gtl",
            "Neighborhood": "NAmes",
            "Condition1": "Feedr",
            "Condition2": "Norm",
            "BldgType": "1Fam",
            "HouseStyle": "1Story",
            "OverallQual": 5,
            "OverallCond": 6,
            "YearBuilt": 1961,
            "YearRemodAdd": 1961,
            "RoofStyle": "Gable",
            "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "None",
            "MasVnrArea": 0.0,
            "ExterQual": "TA",
            "ExterCond": "TA",
            "Foundation": "CBlock",
            "BsmtQual": "TA",
            "BsmtCond": "TA",
            "BsmtExposure": "No",
            "BsmtFinType1": "Rec",
            "BsmtFinSF1": 468,
            "BsmtFinType2": "LwQ",
            "BsmtFinSF2": 144,
            "BsmtUnfSF": 270,
            "TotalBsmtSF": 882,
            "Heating": "GasA",
            "HeatingQC": "TA",
            "CentralAir": "Y",
            "Electrical": "SBrkr",
            "FirstFlrSF": 896,
            "SecondFlrSF": 0,
            "LowQualFinSF": 0,
            "GrLivArea": 896,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
            "FullBath": 1,
            "HalfBath": 0,
            "BedroomAbvGr": 2,
            "KitchenAbvGr": 1,
            "KitchenQual": "TA",
            "TotRmsAbvGrd": 5,
            "Functional": "Typ",
            "Fireplaces": 0,
            "FireplaceQu": "NA",
            "GarageType": "Attchd",
            "GarageYrBlt": 1961.0,
            "GarageFinish": "Unf",
            "GarageCars": 1,
            "GarageArea": 730,
            "GarageQual": "TA",
            "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": 140,
            "OpenPorchSF": 0,
            "EnclosedPorch": 0,
            "ThreeSsnPorch": 0,
            "ScreenPorch": 120,
            "PoolArea": 0,
            "PoolQC": "NA",
            "Fence": "MnPrv",
            "MiscFeature": "NA",
            "MiscVal": 0,
            "MoSold": 6,
            "YrSold": 2010,
            "SaleType": "WD",
            "SaleCondition": "Normal"
        }

        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # print the response
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        return response

# Run the test
test_instance = TestFastAPI()
response = test_instance.test_predict_endpoint()

assert response.status_code == 200
assert "prediction" in response.json()
assert isinstance(response.json()["prediction"], float)
assert response.json()["prediction"] > 0

# Output:
# Status Code: 200
# Response JSON: {'prediction': 4283.597103103181}