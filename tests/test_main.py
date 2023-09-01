import json
import pandas as pd
import os, sys
# sys.path.append('..')
from fastapi.testclient import TestClient

from main import app

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "rasce",
    "sex",
    "native_country",
]

client = TestClient(app)


# test get request
def test_get_greeting_success():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json().get('greeting') == 'Greetings from my API!'

# test post request when data is posted correctly return 200
def test_post_data_success():
    data = {
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
        "native_country": "United-States",
        "salary": ""
        }
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 200

# test post request when data is posted incorrectly return 400
def test_post_data_fail():

    data = {
        "age": 45, 
         "workclass": "State-gov", 
         "fnlgt": 50567,
         "education": "HS-grad", 
         "education_num": 9,
         "marital_status": "Married-civ-spouse",
         "occupation": "Exec-managerial",
         "relationship": "Wife",
         "race": "White",
         "sex": "Female",
         "capital_gain": -10,
         "capital_loss": 0,
         "hours_per_week": 40,
         "native_country": "United-States", 
         "salary": ""
         }

    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 400

# prediction is 1 if salary is >50k
def test_inference_success():

    data = {
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
        "native_country": "United-States",
        "salary": ">50" #giving this salary confirms that predicted salary is correct
    }
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 200

# test getting wrong prediction, prediction is 0 if salary is <=70k
def test_inference_fail():

    data = {
        "age": 45, 
         "workclass": "State-gov", 
         "fnlgt": 50567,
         "education": "HS-grad", 
         "education_num": 9,
         "marital_status": "Married-civ-spouse",
         "occupation": "Exec-managerial",
         "relationship": "Wife",
         "race": "White",
         "sex": "Female",
         "capital_gain": 0,
         "capital_loss": 0,
         "hours_per_week": 40,
         "native_country": "United-States", 
         "salary": '>80' #prediction should be <=50k
         }
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 400


    