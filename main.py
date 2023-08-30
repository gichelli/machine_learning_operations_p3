# Put the code for your API here.
from typing import Union, List
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

# Create a RESTful API using FastAPI this must implement:
class Data(BaseModel):
    feature_1: float
    feature_2: str


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





# Define a POST for the specified endpoint
@app.post("/data/")
async def ingest_data(data: Data):
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