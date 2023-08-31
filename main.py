# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI, HTTPException
from ml.data import process_data
from pydantic import BaseModel
from ml.model import load, inference
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create a RESTful API using FastAPI this must implement:
# class Data(BaseModel):
#     feature_1: float
#     feature_2: str

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
    prediction:int


# get data to test
df = pd.read_csv("data/census.csv", skipinitialspace=True)
df.columns = df.columns.str.replace('-', '_')
train, test = train_test_split(df, test_size=0.20, random_state=42)

# load model, econder and labelbinarizer
model = load(model_path + model_name)
encoder = load(model_path + encoder_name)
lb = load(model_path + lb_name)

# process test data
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)

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


# # Define a POST for the specified endpoint
# @app.post("/data/")
# async def ingest_data1(data: Data1):
#     if data.feature_1 < 0:
#         raise HTTPException(status_code=400, detail="feature_1 needs to be above 0.")
#     if len(data.feature_2) > 280:
#         raise HTTPException(
#             status_code=400,
#             detail=f"feature_2 needs to be less than 281 characters. It has {len(data.feature_2)}.",
#         )
#     return data




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