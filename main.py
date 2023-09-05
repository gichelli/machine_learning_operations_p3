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
    # salary: str
    # prediction:int

    model_config = {
    "json_schema_extra": {
        "examples": [{
                    "age": 29, 
                    "workclass": "Private", 
                    "fnlgt": 185908,
                    "education": "Bachelors", 
                    "education_num": 13,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "Black",
                    "sex": "Male",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 55,
                    "native_country": "United-States",}]
        }
    }

data_greater_than = {
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
    }

data_less_than = {
    "age": 45, 
    "workclass": ["State-gov"], 
    "fnlgt": [50567],
    "education": ["HS-grad"], 
    "education_num": [9],
    "marital_status": ["Married-civ-spouse"],
    "occupation": ["Exec-managerial"],
    "relationship": ["Wife"],
    "race": ["White"],
    "sex": ["Femal"],
    "capital_gain": [0],
    "capital_loss": [0],
    "hours_per_week": [40],
    "native_country": ["United-States"],
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

@app.get('/out/less/')
async def preds():
    data_less_than['salary'] = ''
    pred = get_pred(data_less_than, cat_features, model, encoder, lb)
    return get_label(pred)

@app.get('/out/more/')
async def preds():
    data_greater_than['salary'] = ''
    pred = get_pred(data_greater_than, cat_features, model, encoder, lb)
    return get_label(pred)


# Define a POST for the specified endpoint
#  test cases for EACH of the possible inferences (results/outputs) of the ML model.

@app.post("/data/")
async def update_data(data: Data):
    results = {"age": data.age, 
                "workclass": data.workclass, 
                "fnlgt": data.fnlgt,
                "education": data.education, 
                "education_num": data.education_num,
                "marital_status": data.marital_status,
                "occupation": data.occupation,
                "relationship": data.relationship,
                "race": data.race,
                "sex": data.sex,
                "capital_gain": data.capital_gain,
                "capital_loss": data.capital_loss,
                "hours_per_week": data.hours_per_week,
                "native_country": data.native_country}
    
    # make sure that int values are greater than 0
    for value in results.values():
        if isinstance(value, int) and value < 0:
            raise HTTPException(status_code=400, detail=f"Negative numbers cannot be used.")

    df_data = pd.DataFrame.from_dict([results])

    df_data['salary'] = ''
    pred = get_pred(df_data, cat_features, model, encoder, lb)


    results['salary'] = get_label(pred)

    return results