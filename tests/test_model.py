import pytest
import pandas as pd
import os


@pytest.fixture(scope="session")
def data():
    print("******inside test***********")
    print(os.getcwd())
    # path = '../../data/census.csv'
    path = 'data/census.csv'
    df = pd.read_csv(path, low_memory=False)
    print("-------------")
    return df

def test_data_length(data):
    """
    We test that we have enough data to continue
    """
    assert len(data) > 1000




def test_train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    pass


def test_compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    pass


def test_inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass
