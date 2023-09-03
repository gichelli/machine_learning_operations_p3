'''
Script to POSTS to the API using the requests module and returns both the result 
of model inference and the status code
Author: Gissella Gonzalez
Date: September, 2023
'''
# import libraries
import requests
import json
# Write a script that POSTS to the API using the requests module and returns 
# both the result of model inference and the status code. Include a screenshot
#  of the result. Name this live_post.png.


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

url = 'https://gichelli-a73102c021b3.herokuapp.com/data/'

r = requests.post(url, data=json.dumps(data))
print(r.status_code)
# print(f"Response estatus code: {r.status_code}")
# print(r.json().get('salary'))

# # r = requests.get('https://gichelli-a73102c021b3.herokuapp.com/out/')
# r = requests.get('https://gichelli-a73102c021b3.herokuapp.com/data/')
# print(r.status_code)
# # pred = get_pred(df_data, cat_features, model, encoder, lb)
print(r.json())
# # pred = get
# print(f"Prediction is: {r.json()}")