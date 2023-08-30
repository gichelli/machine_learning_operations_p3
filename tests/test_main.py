import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# test get

def test_get_greeting_success():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == 'Greetings from my API!'


def test_get_greeting_fail():
    r = client.get('/')
    assert r.status_code == 400
    assert r.json() == 56


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
def test_post_data_success():
    data = {"feature_1": 1, "feature_2": "test string"}
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 200


def test_post_data_fail():
    data = {"feature_1": -5, "feature_2": "test string"}
    r = client.post("/data/", data=json.dumps(data))
    assert r.status_code == 400





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