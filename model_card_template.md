# Model Card


## Model Details

This classification model used Census Bureau data to calculate predictions about a persons salary. The salary line of measurement is 50k. Salary can be above or below or equal.

Creation date: Sept 2023
Model version: 1.0.0
Model type: Classification model

The training was done with RandomForestClassifier and parameters where deduced from GridSearchCV. 

Parameters: <br>
max_depth=10<br>
min_samples_split=400<br>
min_samples_leaf=2<br>
criterion='gini'<br>
random_state=42<br>

## Intended Use

This model can be used to predict salary for a given set of values that correspond to an individual.


## Training Data

Fields needed to get a prediction are:

age <br>
workclass <br>
fnlgt <br>
education <br>
education_num<br>
marital_status <br>
occupation<br> 
relationship <br>
race <br>
sex <br>
capital_gain <br>
capital_loss<br>
hours_per_week<br>
native_country <br>

## Evaluation Data

Evaluation data can be found inside out folder. 

slice_output.txt: Includes the Model metrics done in both, the whole dataframe as well as sliced sections of the data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Performance metrics used for this model and values obtained:

precision: 0.7943
recall: 0.5309
fbeta: 0.6364


## Ethical Considerations

Note that the year that the data was obtained: 1994

## Caveats and Recommendations

Data is constantly being updated, therefore salary predictions obtained from this model are accurate for year 1994 which is the year of the Census data used. Data should be updated and model retrained to make sure predictions are accurate at the time of using this model.
