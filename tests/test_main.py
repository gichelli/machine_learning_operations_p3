import json
import pandas as pd
import os, sys
# print(os.getcwd())
sys.path.append('..')
# print(os.getcwd())
from ml.data import process_data
from ml.model import split_and_process_data, load, inference
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from fastapi.testclient import TestClient

from main import app

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
# model_path = '../model/'
# model_name = 'model.pkl'
# encoder_name = 'encoder.pkl'
# lb_name = 'lb.pkl'

client = TestClient(app)



# # # get data to test
# df = pd.read_csv("../data/census.csv", skipinitialspace=True)
# df.columns = df.columns.str.replace('-', '_')
# # # print(df)
# train, test = train_test_split(df, test_size=0.20, random_state=42)
# # # X_train, y_train, encoder, lb = process_data(
# # #         train, categorical_features=cat_features, label="salary", training=True
# # #     )

# # load model, econder and labelbinarizer
# model = load(model_path + model_name)
# encoder = load(model_path + encoder_name)
# lb = load(model_path + lb_name)

# # process test data
# # X_test, y_test, encoder, lb = process_data(test, cat_features, 'salary', True, encoder, lb)

# datax = test.iloc[2]
# print(type(datax))
# print(datax)
# # convert data to json format
# print("------------------")
# dfg = test.apply(lambda x: x.to_json(), axis=1)
# print(type(dfg))

# print(dfg)
# print("------------------")

# print("***************")
# nn= dfg.iloc[2]
# print(nn)
# print("***************")
# print("---***************")
# print("hi")
# # ff = data.to_dict()
# # print(ff)
# print("---***************")

# for i in data.index:
#     print("------------------")
#     print(i)
#     print("------------------")
#     data.loc[i].to_json("row{}.json".format(i))
# print("------------------")

# print(data)

# sample_X_test = X_test[:1]
# pred = inference(model, sample_X_test)

# precision = precision_score(y_test, pred, zero_division=1)
# recall = recall_score(y_test, pred, zero_division=1)
# fbeta = fbeta_score(y_test, pred, beta=1, zero_division=1)

# print(precision)
# print(recall)
# print(fbeta)

# test get
def test_get_greeting_success():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json().get('greeting') == 'Greetings from my API!'


# def test_in_success():
#     data = {"feature_1": 1, "feature_2": "test string"}

#     r = client.post("/predictions/", data=json.dumps(data))
#     assert r.status_code == 200









# def test_get_path():
#     r = client.get("/items/42")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 1 of 42"}


# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200


# test post

# def test_preds_success():
#     r = client.post("/data/", data=json.dumps(nn))
#     print("^^^^^^^^^^^^^^^^^^^^^")
#     print(nn)
#     print(r)
#     print("^^^^^^^^^^^^^^^^^^^^^")
#     assert r.status_code == 200


# # def test_preds_fail():
# json.dumps(nn, indent = 2)

# print(nn)
def test_post_data_success():
    # data = {"feature_1": 1, "feature_2": "test string"}
    # print(nn)
    # data = json.dumps(nn, indent = 2)
    # print(data)
    data = {"age": 29, 
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
         "salary": ">50K",
         'prediction': 1}
    # print(data)
    # print(ff)
    # print(type(p))
    # print(type(ff))
    # print(p)
    # print(ff)
    # ff = data.to_dict()
    # print(ff)
    r = client.post("/data/", data=json.dumps(data))
    # r = client.post("/data/", data={"data":datax.to_json()})

    # r = client.post("/data/", nn)
    print(r.status_code)
    assert r.status_code == 200


# def test_post_data_fail():
#     data = {"feature_1": -5, "feature_2": "test string"}
#     r = client.post("/data/", data=json.dumps(data))
#     assert r.status_code == 400


def test_post_data_fail():

    
    data = {"age": 45, 
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
         "salary": "<=50K",
         'prediction': 1
         }
    # print(data)
    # print(ff)
    # print(type(p))
    # print(type(ff))
    # print(p)
    # print(ff)
    # ff = data.to_dict()
    # print(ff)
    r = client.post("/data/", data=json.dumps(data))
    # r = client.post("/data/", data={"data":datax.to_json()})

    # r = client.post("/data/", nn)
    print(r)
    assert r.status_code == 400


# # test post
# def test_post_data_success():
#     data = {"feature_1": 1, "feature_2": "test string"}
#     r = client.post("/data/", data=json.dumps(data))
#     assert r.status_code == 200


# def test_post_data_fail():
#     data = {"feature_1": -5, "feature_2": "test string"}
#     r = client.post("/data/", data=json.dumps(data))
#     assert r.status_code == 400


# ***********************************************************
# def test_post():
#     data = json.dumps({"value": 10})
#     r = client.post("/42?query=5", data=data)
#     print(r.json())
#     assert r.json()["path"] == 42
#     assert r.json()["query"] == 5
#     assert r.json()["body"] == {"value": 10}

# def test_get_path():
#     r = client.get("/items/42")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 1 of 42"}


# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200



if __name__ == "__main__":
    test_get_greeting_success()
    test_post_data_success()
    # test_post_data_fail()