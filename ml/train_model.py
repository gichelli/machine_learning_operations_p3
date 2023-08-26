# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
from data import process_data
from model import train_model, compute_model_metrics, inference, save_model, sliced_model_metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Add code to load in the data.

data = pd.read_csv("../data/census.csv", skipinitialspace = True)

print(data.head(2))

# get df with encoded columns - use data slicing---maybe
# print("*******************")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
# encoder OneHotEncoder, only used if training=False.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)



# Train and save a model.
model = train_model(X_train, y_train)

# save model
save_model(model)


# Run model inferences and return the predictions.
# X : np.array Data used for prediction.
preds = inference(model, X_test)


# compute_model_metrics(y_test, lr.predict(X_test))
#  y : np.array Known labels, binarized.


# y_test has an issue just by passing it as a parameter
# TypeError: Labels in y_true and y_pred should be of the same type. 
# Got y_true=['<=50K' '>50K'] and y_pred=[0 1]. 
# Make sure that the predictions provided by the classifier coincides with the true labels.



# compute for all the tests in main df
compute_model_metrics(y_test, preds)


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, preds)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

print(matrix)
# # Build the plot
# plt.figure(figsize=(16,7))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix, annot=True, annot_kws={'size':10},
#             cmap=plt.cm.Greens, linewidths=0.2)

# # Add labels to the plot
# # class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
# #                'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
# #                'Krummholz']
# tick_marks = np.arange(len(cat_features))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, cat_features, rotation=25)
# plt.yticks(tick_marks2, cat_features, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.show()

# compute metrics for each slice total 8, cat cols
# workclass below should be passed as a list such as cat_features
# sliced_model_metrics(test, X_test, y_test, 'workclass', model)

# for col in cat_features:
#     sliced_model_metrics(test, X_test, y_test, col, model)


# for each categorical column get a slice of dataframe and calculate metrics
# for col in cat_features:
#     # get slice
#     preds = inference(model, X_test[col])
#     feature_slice(data, y_test, preds, col)
    
#     # feature_slice(data, col, y_test, preds)

#     # compute metrics on slice
#     # compute_model_metrics(y_test, preds)