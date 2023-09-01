# Put the code for your API here.
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from ml.data import process_data
from pydantic import BaseModel
from ml.model import load, inference
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
    age:int 
    workclass:str
    fnlgt:int 
    education:str
    education_num:int 
    marital_status:str
    occupation:str
    relationship:str
    race:str
    sex:str
    capital_gain:int 
    capital_loss:int 
    hours_per_week:int 
    native_country:str
    salary:str


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
    return {'greeting': 'Greetings from my API!'}


# Define a POST for the specified endpoint
#  test cases for EACH of the possible inferences (results/outputs) of the ML model.
@app.post("/data/")
async def ingest_data(data: Data):
    dict_data = dict(data)
    df_data = pd.DataFrame.from_dict([dict_data])

    X_test, _, _, _ = process_data(
    df_data, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

    pred = inference(model, X_test)[0]

    # make sure that int values are greater than 0
    if data.capital_gain or data.capital_loss < 0:
        raise HTTPException(status_code=400, detail="capital_gain and/or capital_loss" 
                            + "needs to be above 0.")

    if df_data['salary'][0] == '>50K'  and pred != 0:
        # pred is 0 when salary is below or equal to 50k
        raise HTTPException(
            status_code=400,
            detail=f"Prediction salary {pred}: is not equal to given salary: {df_data['salary'][0]}",
        )
    if df_data['salary'][0] != '<=50K' and pred == 0:
        # pred is 1 when salary is below or equal to 50k
        raise HTTPException(
            status_code=400,
            detail=f"Prediction salary {pred}: is not equal to given salary: {df_data['salary'][0]}"
        )

    return data







# POST that does model inference.
# Type hinting must be used.
# Use a Pydantic model to ingest the body from POST. This model should contain an example.
# Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
# Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

