# Put the code for your API here.
from typing import Union, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from ml.data import process_data
from pydantic import BaseModel
from ml.model import split_and_process_data, load, inference
from sklearn.model_selection import train_test_split

# Create a RESTful API using FastAPI this must implement:
# class Data(BaseModel):
#     feature_1: float
#     feature_2: str

model_path = 'model/'
model_name = 'model.pkl'
# encoder_name = 'encoder.pkl'
# lb_name = 'lb.pkl'
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

class Data1(BaseModel):
    feature_1: float
    feature_2: str


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
    prediction:int


model_path = 'model/'
model_name = 'model.pkl'
encoder_name = 'encoder.pkl'
lb_name = 'lb.pkl'

# # get data to test
df = pd.read_csv("data/census.csv", skipinitialspace=True)
df.columns = df.columns.str.replace('-', '_')
# # print(df)
train, test = train_test_split(df, test_size=0.20, random_state=42)
# X_train, y_train, encoder, lb = process_data(
#         train, categorical_features=cat_features, label="salary", training=True
#     )

# load model, econder and labelbinarizer
model = load(model_path + model_name)
encoder = load(model_path + encoder_name)
lb = load(model_path + lb_name)
# X_test = load(model_path + 'processed_x_test.pkl')
# y_test = load(model_path + 'processed_y_test.pkl')
# print(model)
# process test data
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)



# df = test.iloc[2]
# print(X_test.iloc[2])
# df1 = test.iloc[2]
# print("^^^^^^^^^^^^^^^^^^^^^^^^^")
# print(train.shape)
# print(test.shape)
# print(X_test.shape)
# print(y_test.shape)


# print("^^^^^^^^^^^^^^^^^^^^^^^^^")

# data={
#         "age": [29], 
#         "workclass": ["Private"], 
#         "fnlgt": [185908],
#         "education": ["Bachelors"], 
#         "education_num": [13],
#         "marital_status": ["Married-civ-spouse"],
#         "occupation": ["Exec-managerial"],
#         "relationship": ["Husband"],
#         "race": ["Black"],
#         "sex": ["Male"],
#         "capital_gain": [0],
#         "capital_loss": [0],
#         "hours_per_week": [55],
#         "native_country": ["United-States"], 
#         "salary": [">50K"]}
    
# df = pd.DataFrame(data)


# variables = Data[0].keys()
# df1 = pd.DataFrame([[getattr(i,j) for j in variables] for i in Data], columns = variables)

# print(df1)


# class Data(BaseModel):
#     data = pd.DataFrame

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


# logging.info('Creating dataframe')
# data = pd.read_csv("data/census.csv", skipinitialspace=True)

# # Optional enhancement, use K-fold cross validation instead of a
# logging.info('Splitting train-test data')
# train, test = train_test_split(data, test_size=0.20)

# # if encoder and lb exist load them, else create them
# logging.info('Checking if model, encoder and labelbinarizer exist')
# check = check_econder_lb(model_path, model_name, encoder_name, lb_name)
# if check == 3:
#     model = load(model_path + model_name)
#     encoder = load(model_path + encoder_name)
#     lb = load(model_path + lb_name)
# else:
#     X_train, y_train, encoder, lb = process_data(
#         train, categorical_features=cat_features, label="salary", training=True
#     )
#     # save encoder and lb
#     logging.info('Saving model, encoder and lb')
#     model = train_model(X_train, y_train)
#     save(model, model_path + model_name)
#     save(encoder, model_path + encoder_name)
#     save(lb, model_path + lb_name)

# Proces the test data with the process_data function.
# # encoder OneHotEncoder, only used if training=False.
# logging.info('processing test data..')
# X_test, y_test, encoder, lb = process_data(
#     test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
# )


# # Run model inferences and return the predictions.
# # X_test : np.array Data used for prediction.
# logging.info('getting predictions..')
# preds = inference(model, X_test)


# preds = model.predict(X)
#     return preds

# @app.post('/data/')
# async def preds_success(data: Data):
#     print("***********inside predictions*********")
#     return {'greeting': 'Greetings from my API!'}

# Define a POST for the specified endpoint
#  test cases for EACH of the possible inferences (results/outputs) of the ML model.
@app.post("/data/")
async def ingest_data(data: Data):
    print("***********inside ingest_data*********")
    print(data.salary)
    # # print(model)
    # # print(df1.head(10))
    # # print(df1.shape)
    # # print(type(data))
    # print(data)
    # pred = model.predict(df)
    # print(pred)
    # df = pd.read_json(data)
    # print(df)
    # print("***********inside ingest_data*********")
    pred = inference(model, X_test)
    # print(pred)
    # print(pred[2])
    print(pred[1])
    if data.salary == '>50K' and data.prediction != pred[2]:
    # pred is 1 when salary is above 50k
        print("greater than 50")
        raise HTTPException(status_code=400, detail="write reason of error here")
    if data.salary == '<=50K' and data.prediction != pred[1]:
        print("less than 50k")
        # pred is 1 when salary is below or equal to 50k
        raise HTTPException(
            status_code=400,
            detail="write reason of error here"
        )
    return data


# Define a POST for the specified endpoint
@app.post("/data/")
async def ingest_data1(data: Data1):
    if data.feature_1 < 0:
        raise HTTPException(status_code=400, detail="feature_1 needs to be above 0.")
    if len(data.feature_2) > 280:
        raise HTTPException(
            status_code=400,
            detail=f"feature_2 needs to be less than 281 characters. It has {len(data.feature_2)}.",
        )
    return data




# POST that does model inference.
# Type hinting must be used.
# Use a Pydantic model to ingest the body from POST. This model should contain an example.
# Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
# Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).





# ***********************************************************


# class Value(BaseModel):
#     value: int

# # Use POST action to send data to the server
# @app.post("/{path}")
# async def exercise_function(path: int, query: int, body: Value):
#     return {"path": path, "query": query, "body": body}


# @app.get("/items/{item_id}")
# async def get_items(item_id: int, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}