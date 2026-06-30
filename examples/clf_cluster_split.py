import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from imblearn.over_sampling import SMOTE
import os

# change these two accordingly
biological_target = 'hek'
measurement = 'ic50(mm)'

data = pd.read_csv(f'{biological_target}_{measurement}_mixture_descriptors_alvadesc_mordred.csv')



def split_x_y(data):
  X = data.drop(columns=['value'])
  y = data['value']
  return X, y


def prepare_data(data):
    drop_list = ['sample']
    data = data.drop(columns=drop_list)
    # column values: if it is a number --> 1, else --> 0
    data['value'] = data['value'].apply(lambda x: 0 if x == 'ni' else 1)
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
if not os.path.exists('classification'):
  os.makedirs('classification')
# save the data
X_train.to_csv('classification/X_train.csv', index=False)
X_test.to_csv('classification/X_test.csv', index=False)
y_train.to_csv('classification/y_train.csv', index=False)
y_test.to_csv('classification/y_test.csv', index=False)

print(f"number of features in X_train after preprocessing: {X_train.shape[1]}")
print(f"number of features in X_test after preprocessing: {X_test.shape[1]}")

print(f"number of train data: {X_train.shape[0]}")
print(f"number of test data: {X_test.shape[0]}")

print(f"number of positive data: {y_train.sum()}")
print(f"number of negative data: {len(y_train) - y_train.sum()}")
print(f"number of positive data: {y_test.sum()}")
print(f"number of negative data: {len(y_test) - y_test.sum()}")

# since data is imbalanced, we need to balance the data
# we can use the SMOTE algorithm to balance the data
smote = SMOTE(random_state=42, sampling_strategy='minority', k_neighbors=3)
X_train, y_train = smote.fit_resample(X_train, y_train)

# # save the balanced data
X_train.to_csv('classification/X_train_balanced.csv', index=False)
y_train.to_csv('classification/y_train_balanced.csv', index=False)

print(f"number of positive data in balanced data: {y_train.sum()}")
print(f"number of negative data in balanced data: {len(y_train) - y_train.sum()}")
print(f"type y_train: {type(y_train)}")