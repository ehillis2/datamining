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


def DTmodelScore(model, trainingdata, traininglabels, testingdata, testinglabels):
    model.fit(trainingdata, traininglabels)
    return model.score(testingdata, testinglabels)


def DecisionTree(data_features, data_labels, decisiontreemodel):

    folds = StratifiedKFold(n_splits=10)

    scores = []

    for train_index, test_index in folds.split(data_features, data_labels):
        X_train, X_test, y_train, y_test = data_features.iloc[train_index], data_features.iloc[test_index], \
            data_labels.iloc[train_index], data_labels.iloc[test_index]
        scores.append(100 * DTmodelScore(decisiontreemodel,
                                         X_train, y_train, X_test, y_test))

    #print('these are the scores: ', scores)
    # print('mean:', np.mean(scores))
    return np.mean(scores)


DT_STD_means = []
DT_PCA_means = []


def mainDecisionTreeImplementation(decisiontreemodel, linPCA, training_dataframe, pca_option):

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

    if(pca_option == 'no'):
        # print('model without PCA')
        DT_STD_means.append(DecisionTree(
            features_data, labels_data, decisiontreemodel))

    elif(pca_option == 'both'):

        DT_STD_means.append(DecisionTree(
            features_data, labels_data, decisiontreemodel))

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        # print('Decision Tree model with PCA')

        DT_PCA_means.append(DecisionTree(
            features_df_PCA, labels_data, decisiontreemodel))
    else:
        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        # print('Decision Tree model with PCA')

        DT_PCA_means.append(DecisionTree(
            features_df_PCA, labels_data, decisiontreemodel))

    # print()


# mainDecisionTreeImplementation()

def DecisionTreeSimulation(decisiontreemodel, linPCA, training_dataframe, pca_option):

    if(pca_option == 'both'):
        number = 20
        print('Simulating decision tree model...')
        start = time.time()
        for i in range(0, number):
            # print('decision tree simulation number', i, 'finished')
            mainDecisionTreeImplementation(
                decisiontreemodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Decsion Tree Simulation time:', end - start)

        m = None

        if np.mean(DT_STD_means) > np.mean(DT_PCA_means):
            m = 'STANDARD MODEL'
        else:
            m = 'PCA TRANSFORMED MODEL'

        count = 0

        for i in range(0, number):
            if(DT_PCA_means[i] > DT_STD_means[i]):
                count = count + 1

        print()
        print('number of times pca transformed decision tree model had greater accuracy than standard decision tree model: ', count, 'out of ', number)
        print('decision tree variance in accuracies for standard model: ',
              np.var(DT_STD_means))
        print('decision tree variance in accuracies for pca transformed model: ',
              np.var(DT_PCA_means))
        print('decision tree standard model accuracy: ', np.mean(DT_STD_means))
        print('decision tree pca transformed model accuracy: ',
              np.mean(DT_PCA_means))
        print('maximum decision tree accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
            np.mean(DT_PCA_means), np.mean(DT_STD_means)), '%')

        print()
    elif(pca_option == 'yes'):
        number = 20
        print('Simulating PCA transformed decision tree model...')
        start = time.time()
        for i in range(0, number):
            # print('decision tree simulation number', i, 'finished')
            mainDecisionTreeImplementation(
                decisiontreemodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Decsion Tree Simulation time:', end - start)

        print()

        print('pca transformed decision tree variance in accuracies: ',
              np.var(DT_PCA_means))
        print('decision tree pca transformed model accuracy on 10fold cross-val test data: ',
              np.mean(DT_PCA_means))
        print()
    else:
        number = 20
        print('Simulating PCA transformed decision tree model...')
        start = time.time()
        for i in range(0, number):
            # print('decision tree simulation number', i, 'finished')
            mainDecisionTreeImplementation(
                decisiontreemodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Decsion Tree Simulation time:', end - start)

        print()

        print('standard decision tree variance in accuracies: ',
              np.var(DT_STD_means))
        print('standard decision tree model accuracy on 10fold cross-val test data: ',
              np.mean(DT_STD_means))
        print()


dt = DecisionTreeClassifier()
"""
start = time.time()
mainDecisionTreeImplementation(dt, processing.linear_pca, processing.overall_training_data)
end = time.time()
print("Time of mainDecisionTreeImplementation", end - start)
"""
# DecisionTreeSimulation(dt, processing.linear_pca,
#                       processing.overall_training_data)
