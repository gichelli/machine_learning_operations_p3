'''
Script to train machine learning model.

Author: Gissella Gonzalez
Date  : August 28, 2023
'''
# import libraries
import os, logging
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import *
from ml.data import process_data


# create logging
logging.basicConfig(
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename='logs/train_model.log',
    filemode='w+',
    level=logging.INFO
)

# import logging
logging.getLogger().addHandler(logging.StreamHandler())

# Add code to load in the data.
# list of categorical columns
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

# path to model, encoder and lb
model_path = 'model/'

logging.info('Creating dataframe')
data = pd.read_csv("data/census.csv", skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a
logging.info('Splitting train-test data')
train, test = train_test_split(data, test_size=0.20)

# if encoder and lb exist load them, else create them
logging.info('Checking if encoder and labelbinarizer exist')
check = check_econder_lb(model_path)
if check == 2:
    encoder, lb = get_encoder_lb(model_path)
else:
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    # save encoder and lb
    logging.info('Saving encoder and lb')
    save_encoder_and_lb(model_path, encoder, lb)

# if model exist get load it
logging.info('If model exists, loading it.. else training and saving it..')
if os.path.isfile(model_path + 'model.pkl'):
    model = load_model(model_path + 'model.pkl')

# else train and save model
else:
    model = train_model(X_train, y_train)
    save_model(model, model_path + 'model.pkl')

# load encoder and lb
logging.info('loading encoder and lb..')
encoder, lb = get_encoder_lb(model_path)


# Proces the test data with the process_data function.
# encoder OneHotEncoder, only used if training=False.
logging.info('precessing test data..')
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)


# Run model inferences and return the predictions.
# X_test : np.array Data used for prediction.
logging.info('getting predictions..')
preds = inference(model, X_test)

# compute for all the tests in main df
logging.info('computing metrics for all data..')
compute_model_metrics(y_test, preds)

# Get and reshape confusion matrix data
# matrix = confusion_matrix(y_test, preds)
# print(matrix)

logging.info('Computing metrics for each categorical column slice..')
'''
compute metrics for each slice total 8, cat cols
workclass below should be passed as a list such as cat_features
sliced_model_metrics(test, X_test, y_test, 'workclass', model)
'''
out_path = 'out/'
if os.path.isfile(out_path + 'slice_output.txt'):
    os.remove(out_path + 'slice_output.txt')
for col in cat_features:
    sliced_model_metrics(test, X_test, y_test, col, model)
logging.info('done.')