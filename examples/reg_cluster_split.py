import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os

# change these two accordingly
biological_target = 'hek'
measurement = 'ic50(mm)'

data = pd.read_csv(f'{biological_target}_{measurement}_mixture_descriptors_alvadesc_mordred.csv')
# note: do not confuse the above file with 
#regression_ecoli_distance_alvadesc_mordred.csv I just saved it
# to make sure the file is correct. this mixture descriptors file 
# is the one that is used in both regression and classification.



def split_x_y(data):
  X = data.drop(columns=['value'])
  y = data['value']
  return X, y


def prepare_data(data):
    drop_list = ['sample']
    data = data.drop(columns=drop_list)
    # column values: if it is a number --> number, else --> remove
    data['value'] = data['value'].apply(lambda x: 0 if x == 'ni' else x)
    # now drop the rows where value is 0
    data = data[data['value'] != 0]
    # the values are string and should use eval to convert to float
    data['value'] = data['value'].apply(lambda x: eval(x))
    return data

def preprocess_data(X_train, X_test):
  X_train, X_test = remove_constant_features(X_train, X_test)
  X_train, X_test = remove_highly_correlated_features(X_train, X_test)
  X_train, X_test = remove_low_variance_features(X_train, X_test)
  return X_train, X_test



data = prepare_data(data)
data.to_csv(f'regression_{biological_target}_{measurement}_alvadesc_mordred.csv', index=False)

# split the data
data_x, data_y = split_x_y(data)
# do umap clustering on X data 
umap_df = umap_clustering(data_x)
# split the data into train and test for each cluster
X_train, X_test, y_train, y_test = cluster_split(umap_df, data_x, data_y, plot=True, verbose=False)
# preprocess the data
X_train, X_test = normalize_data(X_train, X_test)
X_train, X_test = preprocess_data(X_train, X_test)

# create a folder to save the data
if not os.path.exists('regression'):
  os.makedirs('regression')
# save the data
X_train.to_csv('regression/X_train.csv', index=False)
X_test.to_csv('regression/X_test.csv', index=False)
y_train.to_csv('regression/y_train.csv', index=False)
y_test.to_csv('regression/y_test.csv', index=False)

print(f"number of features in X_train after preprocessing: {X_train.shape[1]}")
print(f"number of features in X_test after preprocessing: {X_test.shape[1]}")

print(f"number of train data: {X_train.shape[0]}")
print(f"number of test data: {X_test.shape[0]}")
