'''
this function computes performance metrics of the model holding a fixed feature
(slices of the data)
'''
import pandas as pd


'''
Write a function that computes performance on model slices. I.e. a function that computes 
the performance metrics when the value of a given feature is held fixed. E.g. for education, 
it would print out the model metrics for each slice of data that has a particular
 value for education. You should have one set of outputs for every single unique value in education.
'''

# Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.

#categorical features: 
# ['workclass', 'education', ', 'occupation', 'relationship', 'race', 'sex',  'native-country']



df = pd.read_csv('data/census.csv')
print(df.head(2))


#replace hiphen for underscore in column names
df.columns = df.columns.str.strip().str.replace('-', '_')
print(df.head(2))

def slice_census(df, feature):
    #calculate metrics
    print()

slice_census(df, 'workclass')
slice_census(df, 'education')
slice_census(df, 'marital-status')
slice_census(df, 'occupation')
slice_census(df, 'relationship')
slice_census(df, 'race')
slice_census(df, 'sex')
slice_census(df, 'native-country')