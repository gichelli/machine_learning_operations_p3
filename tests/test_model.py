import pytest
import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys
print(os.getcwd())
sys.path.append('..')
print(os.getcwd())
print(sys.path)
from ml.data import process_data
from ml.model import train_model, inference, save_model, load_model, sliced_model_metrics

# path = os.path.dirname(os.getcwd())

# logging.basicConfig(
#     format='%(name)s - %(levelname)s - %(message)s',
#     datefmt='%m/%d/%Y %I:%M:%S %p',
#     filename='./logs/model.log',
#     filemode='w+',
#     level=logging.INFO
# )
model_path = 'model/'
out_path = 'out/'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="session")
def data():
    print(os.getcwd())
    # path = '../../data/census.csv'
    path = 'data/census.csv'
    df = pd.read_csv(path, skipinitialspace = True)
    # print("-------------")
    return df

@pytest.fixture(scope="session")
def get_model():
    # path = os.getcwd()
    # path = os.path.dirname(os.getcwd())
    # current_dirs_parent = os.path.dirname(os.getcwd())
    print("**********************--------------------")
    # print(path)
    # print(current_dirs_parent)
    # print("**********************--------------------")
    # path = 'model/'
    # x = ''
    print(os.getcwd())
    print("**********************--------------------")

    model = load_model(model_path + 'model_sample.pkl')
    return model

@pytest.fixture(scope="session")
def data_split(data):
    # train, test = split_data(data)
    train, test = train_test_split(data, test_size=0.20)
    return train, test


# def test_data_length(data):
#     """
#     We test that we have enough data to continue
#     """
#     assert len(data) > 1000



@pytest.fixture(scope="module")
def get_train_data(data_split):
    '''
    Return trained data
    '''
    # train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
    data_split[0], cat_features, label="salary", training=True
)

    return X_train, y_train, encoder, lb

@pytest.fixture(scope="module")
def get_test_data(data_split, get_train_data):
    '''
    Return tested data
    '''
    # test, test = test_test_split(data, test_size=0.20)
    
    X_test, y_test, encoder, lb = process_data(
    data_split[1], categorical_features=cat_features, label="salary", training=False, encoder=get_train_data[2], lb=get_train_data[3])

    return X_test, y_test
    


def test_save_model(get_train_data):
    '''
    test to see if function is saving models correctly
    '''

    try:
        model_name = '/model_sample.pkl'
        model_sample = train_model(get_train_data[0], get_train_data[1])
        save_model(model_sample, model_path + model_name)
       
        assert os.path.isfile(model_path + model_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing save_model - model has not been saved in model folder")
        raise err
    

#  save_encoder_and_lb(model_path, encoder, lb)
def test_save_encoder_and_lb(get_train_data):
    '''
    test to see if function is saving encoder and lb correctly
    '''
    encoder_name = '/model_sample.pkl'
    lb_name = '/lb_sample.pkl'
    try:
        save_model(get_train_data[2], model_path + encoder_name)
        assert os.path.isfile(model_path + encoder_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_save_encoder_and_lb - encoder has not been saved in model folder")
        raise err
    
    try:
        save_model(get_train_data[3], model_path + lb_name)
        assert os.path.isfile(model_path + lb_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_save_encoder_and_lb - lb has not been saved in model folder")
        raise err
    


def test_inference(data, get_model, get_test_data):
    '''
    test prediction's type
    '''
    try:
        preds = inference(get_model, get_test_data[0])
        assert isinstance(preds, np.ndarray)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_inference - preds is not a np.array")
        raise err

    # test size of predictions
    try:
        assert len(preds) == get_test_data[0].shape[0]
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_inference - preds is not the same size as test data")
        raise err
    


def test_sliced_model_metrics(data_split, get_test_data, get_model, data):
    '''
    test to check if function calculates metrics for each class in each feature
    number of features = len(cat_feaures)
    number of unique classes in each feature 

    '''
    num_classes = len(data['workclass'].unique())
    print(num_classes)
    try:
        assert sliced_model_metrics(
            data_split[1], get_test_data[0], get_test_data[1], 'workclass', get_model, total_count=[]
            ) == num_classes
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_sliced_model_metrics - did not run exact lenght of cat_features")
        raise err
   


# def test_compute_model_metrics(y, preds):
def test_compute_model_metrics():
    """
    test to see if slice_output file is not a blank file
    """
    try:
        assert os.stat(out_path + 'slice_output.txt').st_size != 0 
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_compute_model_metrics - there is not a file named slice_output.txt in out folder")
        raise err
    


