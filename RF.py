import processing

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
import time


def RFmodelScore(model, trainingdata, traininglabels, testingdata, testinglabels):
    model.fit(trainingdata, traininglabels.values.ravel())
    return model.score(testingdata, testinglabels)

def RandomForest(data_features, data_labels, randomforestmodel):

    folds = StratifiedKFold(n_splits=10)
    # m = RandomForestClassifier()

    scores = []

    for train_index, test_index in folds.split(data_features, data_labels):
        X_train, X_test, y_train, y_test = data_features.iloc[train_index], data_features.iloc[test_index], \
            data_labels.iloc[train_index], data_labels.iloc[test_index]
        scores.append(100 * RFmodelScore(randomforestmodel,
                                         X_train, y_train, X_test, y_test))

    #print('these are the scores: ', scores)
    #print('mean:', np.mean(scores))
    return np.mean(scores)


RF_STD_means = []
RF_PCA_means = []

def mainRandomForestImplementation(randomforestmodel, linPCA, training_dataframe, pca_option):

    df = pd.read_csv("BreastCancerData.csv")
    column_titles = None
    column_titles = processing.processData(df, column_titles)
    column_titles = np.array(column_titles)

    training_data = []
    final_validation = []

    processing.splitData(training_dataframe, .8,
                         training_data, final_validation)

    training_data = np.array(training_data)
    final_validation = np.array(final_validation)

    features = []
    labels = []

    features, labels = processing.createFeatures_Labels(training_data)
    np.transpose(labels)

    features_data = None
    labels_data = None

    features_data, labels_data = processing.convertToDataFrame(
        features, labels, column_titles)

    if(pca_option == 'both'):

        #print('Random Forest model without PCA')
        RF_STD_means.append(RandomForest(
            features_data, labels_data, randomforestmodel))
        # print()

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('Random Forest model with PCA')
        RF_PCA_means.append(RandomForest(
            features_df_PCA, labels_data, randomforestmodel))
        # print()

    elif(pca_option == 'yes'):

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('Random Forest model with PCA')
        RF_PCA_means.append(RandomForest(
            features_df_PCA, labels_data, randomforestmodel))

    else:

        RF_STD_means.append(RandomForest(
            features_data, labels_data, randomforestmodel))


def RandomForestSimulation(randomforestmodel, linPCA, training_dataframe, pca_option):

    print()
    if(pca_option == 'both'):

        number = 20
        print('Simulating random forest models...')
        start = time.time()
        for i in range(0, number):
            #print('random forest simulation number', i, 'finished')
            mainRandomForestImplementation(
                randomforestmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Random Forest Simulation time:', end - start)

        m = None

        if np.mean(RF_STD_means) > np.mean(RF_PCA_means):
            m = 'STANDARD MODEL'
        else:
            m = 'PCA TRANSFORMED MODEL'

        count = 0

        for i in range(0, number):
            if(RF_PCA_means[i] > RF_STD_means[i]):
                count = count + 1

        print()
        print('number of times random forest pca transformed model had greater accuracy than random forest standard model: ',
              count, 'out of ', number)
        print('random forest variance in accuracy for standard model: ',
              np.var(RF_STD_means))
        print('random forest variance in accuracy for pca transform model: ',
              np.var(RF_PCA_means))
        print('random forest standard model accuracy on 10fold cross-val test data: ',
              np.mean(RF_STD_means))
        print('random forest pca transformed model accuracy on 10fold cross-val test data: ',
              np.mean(RF_PCA_means))
        print('maximum random forest accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
            np.mean(RF_PCA_means), np.mean(RF_STD_means)), '%')
        print()

    elif(pca_option == 'yes'):

        number = 20
        print('Simulating PCA transformed random forest model...')
        start = time.time()
        for i in range(0, number):
            #print('random forest simulation number', i, 'finished')
            mainRandomForestImplementation(
                randomforestmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Random Forest Simulation time:', end - start)

        print()
        print('pca transformed random forest model variance in accuracy: ',
              np.var(RF_PCA_means))
        print('pca transformed random forest model accuracy on 10fold cross-val test data: ',
              np.mean(RF_PCA_means), '%')
        print()

    else:

        number = 20
        print('Simulating standard random forest model...')
        start = time.time()
        for i in range(0, number):
            #print('random forest simulation number', i, 'finished')
            mainRandomForestImplementation(
                randomforestmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Random Forest Simulation time:', end - start)

        print()
        print('standard random forest variance in accuracy: ',
              np.var(RF_STD_means))
        print('standard random forest accuracy on 10fold cross-val test data: ',
              np.mean(RF_STD_means))
        print()


rf = RandomForestClassifier()
"""
start = time.time()
mainRandomForestImplementation(
    rf, processing.linear_pca, processing.overall_training_data)
end = time.time()
print('Time of mainRandomForestImplementation', end - start)
"""
# RandomForestSimulation(rf, processing.linear_pca,
#                       processing.overall_training_data)
