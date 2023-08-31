'''
Script to define model functions

Author: Gissella Gonzalez
Date  : August 28, 2023
'''
# import libraries
import os
import pickle
from ml.data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


out_path = 'out/'

def split_and_process_data(train, cat_features, label, training):
    X, y, encoder, lb = process_data(
        train, cat_features, label, training
    )

    return X, y, encoder, lb


def check_econder_lb(path, model_name, encoder_name, lb_name):
    '''
    check if encoder or LabelBinarizer already exists
    input: path of obkect locations
    output: returns int that ensures both object exists in path
    '''
    names = [model_name, encoder_name, lb_name]
    count = 0
    for file in names:
        if os.path.isfile(path + file):
            count += 1
    return count


def load(path):
    '''
    load object from path
    input: path of object locations
    output: returns object
    '''
    return pickle.load(open(path, 'rb'))


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
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
    rfc = RandomForestClassifier(
        max_depth=10,
        min_samples_split=400,
        min_samples_leaf=2,
        criterion='gini',
        random_state=42)
    rfc.fit(X_train, y_train)

    return rfc


# save objects
def save(obj, path):
    '''
    save obj in given path
    input: path of object locations

    '''
    pickle.dump(obj, open(path, 'wb'))


def inference(model, X):
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
    preds = model.predict(X)
    return preds


def sliced_model_metrics(df, X_test, y_test, feature, model):
    '''
    Function to compute model metrics on slices of the dataset
    input:
    df - splited test dataframe
    X_test - processed_data test numpy.ndarray
    y_test - processed_data labels values numpy.ndarray
    feature - column name
    model
    '''
    count = 0
    with open(out_path + 'slice_output.txt', 'a', encoding="utf8") as file:
        file.write('####################################################\n')
        file.writelines('Performance metrics for ' + feature + ' slice:\n')
        file.write('####################################################\n')
        file.write('\n')

    for cls in df[feature].unique():
        count += 1
        row_slice = df[feature] == cls
        preds = inference(model, X_test[row_slice])

        with open(out_path + 'slice_output.txt', 'a', encoding="utf8") as file:
            file.writelines('Computer metrics for ' + cls + ' class\n')

        compute_model_metrics(y_test[row_slice], preds)
    return count


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    prints metrics to slice_output.txt file
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

    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    with open(out_path + 'slice_output.txt', 'a', encoding="utf8") as file:
        file.writelines(f'precision: {precision:.4f}\n')
        file.writelines(f'recall: {recall:.4f}\n')
        file.writelines(f'fbeta: {fbeta:.4f}\n')
        file.write('\n')
        file.close()

    return precision, recall, fbeta
