from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import operator
import sys
import math
import csv


df = pd.read_csv("BreastCancerData.csv")


column_titles = None


def processData(data, column_array):
    del data['id']
    data = data.dropna()
    column_array = list(data.columns)
    return column_array


column_titles = processData(df, column_titles)
column_titles = np.array(column_titles)

training_data = []
final_validation = []

def splitData(dataset, split, trainingdata, finalvalidation):

    split = 1 - split

    for i in range(len(dataset)):
        if(random.random() < split):
            finalvalidation.extend([dataset.iloc[i]])
        else:
            trainingdata.extend([dataset.iloc[i]])

    finalvalidation = np.array(finalvalidation)
    trainingdata = np.array(trainingdata)


splitData(df, .9, training_data, final_validation)

overall_training_data = pd.DataFrame(training_data, columns=column_titles)
overall_final_validation = pd.DataFrame(
    final_validation, columns=column_titles)


features = []
labels = []

def createFeatures_Labels(data):

    features = data[:, 1:]
    labels = data[:, 0]
    return features, labels


def convertToDataFrame(f_arr, l_arr, col_names):
    c = col_names[1:]
    features_data = pd.DataFrame(f_arr, columns=c)
    lab = col_names[0]
    labels_data = pd.DataFrame(l_arr, columns=['class'])
    return features_data, labels_data


linear_pca = PCA(n_components=.9, svd_solver='full')


def linearPCAReduction(dataframe, pca_used):
    #dataframe.subtract(dataframe.mean())
    return pca_used.fit_transform(dataframe)
