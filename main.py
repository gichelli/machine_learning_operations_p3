'''
Script to define API data ingestion

Author: Gissella Gonzalez
Date  : Sept, 2023
'''
# import libraries
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from ml.data import process_data
from pydantic import BaseModel
from ml.model import load, inference, get_label, get_pred
from sklearn.model_selection import train_test_split

# Create a RESTful API using FastAPI this must implement:

model_path = 'model/'
model_name = 'model.pkl'
encoder_name = 'encoder.pkl'
lb_name = 'lb.pkl'

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
model = load(model_path + model_name)


class Get(BaseModel):
    greeting: str 
    prediction: str

class Data(BaseModel):
    age: int 
    workclass: str
    fnlgt: int 
    education: str
    education_num: int 
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: str
    salary: str
    # prediction:int


data = {
    "age": 29, 
    "workclass": ["Private"], 
    "fnlgt": [185908],
    "education": ["Bachelors"], 
    "education_num": [13],
    "marital_status": ["Married-civ-spouse"],
    "occupation": ["Exec-managerial"],
    "relationship": ["Husband"],
    "race": ["Black"],
    "sex": ["Male"],
    "capital_gain": [0],
    "capital_loss": [0],
    "hours_per_week": [55],
    "native_country": ["United-States"],
    "salary": [">50K"]
    }

# get data to test
df = pd.read_csv("data/census.csv", skipinitialspace=True)
df.columns = df.columns.str.replace('-', '_')
train, test = train_test_split(df, test_size=0.20, random_state=42)

# load model, econder and labelbinarizer
model = load(model_path + model_name)
encoder = load(model_path + encoder_name)
lb = load(model_path + lb_name)

app = FastAPI(
    title="Gretting API",
    description="An API that gives a welcome messages and post model inference",
    version="1.0.0",
)


# GET on the root giving a welcome message.
# Define a GET for the specified endpoint
@app.get('/')
async def greet():
    return 'Greetings from my API!'


@app.get('/out/')
async def preds():
    pred = get_pred(data, cat_features, model, encoder, lb)
    return get_label(pred)



# Define a POST for the specified endpoint
#  test cases for EACH of the possible inferences (results/outputs) of the ML model.
@app.post("/data/")
async def ingest_data(data: Data):
    dict_data = dict(data)
    df_data = pd.DataFrame.from_dict([dict_data])

    pred = get_pred(df_data, cat_features, model, encoder, lb)

    # make sure that int values are greater than 0
    if data.capital_gain or data.capital_loss < 0:
        raise HTTPException(status_code=400, detail="capital_gain and/or capital_loss" 
                            + "needs to be above 0.")

    if df_data['salary'][0] == '>50K'  and pred != 0:
        # pred is 0 when salary is below or equal to 50k
        raise HTTPException(status_code=400,
                            detail=f"Prediction salary {pred}: is not equal to given salary: {df_data['salary'][0]}",
        )
    if df_data['salary'][0] != '<=50K' and pred == 0:
        # pred is 1 when salary is below or equal to 50k
        raise HTTPException(status_code=400,
                            detail=f"Prediction salary {pred}: is not equal to given salary: {df_data['salary'][0]}"
        )

    return data
