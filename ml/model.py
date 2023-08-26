from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


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
    print("************inside train_model function")
    rfc = RandomForestClassifier(max_depth=10, min_samples_split=400, min_samples_leaf=2, criterion='gini',random_state=42)
    rfc.fit(X_train, y_train)

    return rfc


#save the model
def save_model(model):
    joblib.dump(model, '../model/model.pkl')
     

# lead the model
def load_model():
    pass


# main instruction ---Write a function that computes model metrics on slices of the data.

# Write a function that computes performance on model slices. 
# I.e. a function that computes the performance metrics when the value of a given 
# feature is held fixed. E.g. for education, it would print out the model metrics 
# for each slice of data that has a particular value for education. You should have
# one set of outputs for every single unique value in education.



# Complete the stubbed function or write a new one that for a given categorical 
# variable computes the metrics when its value is held fixed.


# Write a script that runs this function (or include it as part of the training script) 
# that iterates through the distinct values in one of the features and prints out the 
# model metrics for each value.



# Output the printout to a file named slice_output.txt.





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

# def feature_slice(df, y_test, preds, col):
#     '''
#     Function to compute model metrics on slices of the dataset
#     '''

#     for cls in df[col].unique():
#         print(cls)

#         compute_model_metrics(y_test, preds, cls)
#         # df_temp = df[df["class"] == cls]
#         # mean = df_temp[col].mean()
#         # stddev = df_temp[col].std()
#         # print(f"Class: {cls}")
#         # print(f"{col} mean: {mean:.4f}")
#         # print(f"{col} stddev: {stddev:.4f}")
#     # print()

def sliced_model_metrics(df, X_test, y_test, feature, model):
    print("Performance metrics for " + feature + " slice:\n")
    for cls in df[feature].unique():
        row_slice = df[feature] == cls
        
        preds = inference(model, X_test[row_slice])
        print("Computer metrics for " + cls + " class")
        
        compute_model_metrics(y_test[row_slice], preds)
#         print(x)
        
#     return x
        
#     return x
# slice_dataframe(df, "sepal_length")
# slice_iris(df, "sepal_width")
# slice_iris(df, "petal_length")
# slice_iris(df, "petal_width")


def compute_model_metrics(y, preds):
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
    #     print("************--->inside compute_model_metrics ****************")
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"fbeta: {fbeta:.4f}\n")
#     print()
    return precision, recall, fbeta
