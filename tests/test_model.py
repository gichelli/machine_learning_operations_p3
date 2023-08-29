'''
Script to test model functions

Author: Gissella Gonzalez
Date  : August 28, 2023
'''
# import libraries
import os
import sys
import pandas as pd
import numpy as np
import pytest
import logging
from sklearn.model_selection import train_test_split
from ml.model import train_model, inference, save, load, sliced_model_metrics
from ml.data import process_data


model_path = 'model/'
out_path = 'out/'
model_name = 'model_sample.pkl'
encoder_name = 'encoder_sample.pkl'
lb_name = 'lb_sample.pkl'

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
    '''
    fixture - get dataframe
    '''
    print(os.getcwd())
    path = 'data/census.csv'
    df = pd.read_csv(path, skipinitialspace=True)
    return df


@pytest.fixture(scope="session")
def get_model():
    '''
    fixture - get model
    '''
    model = load(model_path + model_name)
    return model


@pytest.fixture(scope="session")
def data_split(data):
    '''
    fixture - split data to test and train datasets
    '''
    train, test = train_test_split(data, test_size=0.20)
    return train, test


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
    X_test, y_test, encoder, lb = process_data(
        data_split[1], categorical_features=cat_features, label="salary", training=False, encoder=get_train_data[2], lb=get_train_data[3])

    return X_test, y_test


def test_save(get_train_data):
    '''
    test to see if function is saving models correctly
    '''
    try:
        model_sample = train_model(get_train_data[0], get_train_data[1])
        save(model_sample, model_path + model_name)
        assert os.path.isfile(model_path + model_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_save - model has not been saved in model folder")
        raise err

    try:
        save(get_train_data[2], model_path + encoder_name)
        assert os.path.isfile(model_path + encoder_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_save - encoder has not been saved in model folder")
        raise err

    try:
        save(get_train_data[3], model_path + lb_name)
        assert os.path.isfile(model_path + lb_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_save - lb has not been saved in model folder")
        raise err


def test_load():
    '''
    test loading encoder and lb is working as expected
    '''
    try:
        if os.path.isfile(model_path + model_name):
            assert load(model_path + model_name)
    except AssertionError as err:
        logging.error(
            "ERROR: Testing test_load - model has not loaded properly")
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
            data_split[1],
            get_test_data[0],
            get_test_data[1],
            'workclass',
            get_model,
            total_count=[]) == num_classes
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
