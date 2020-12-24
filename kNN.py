import processing
#from FinalValidationTest import option

from sklearn.neighbors import KNeighborsClassifier
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


def kNNmodelScore(model, trainingdata, traininglabels, testingdata, testinglabels):
    model.fit(trainingdata, traininglabels.values.ravel())
    return model.score(testingdata, testinglabels)


def kNearestNeighbor(data_features, data_labels, knnmodel):

    folds = StratifiedKFold(n_splits=10)

    scores = []

    for train_index, test_index in folds.split(data_features, data_labels):
        X_train, X_test, y_train, y_test = data_features.iloc[train_index], data_features.iloc[test_index], \
            data_labels.iloc[train_index], data_labels.iloc[test_index]
        scores.append(100 * kNNmodelScore(knnmodel,
                                          X_train, y_train, X_test, y_test))

    #print('these are the scores: ', scores)
    #print('mean:', np.mean(scores))
    return np.mean(scores)


kNN_STD_means = []
kNN_PCA_means = []


def mainkNearestNeighborImplementation(knnmodel, linPCA, training_dataframe, pca_option):

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
        kNN_STD_means.append(kNearestNeighbor(
            features_data, labels_data, knnmodel))

    elif(pca_option == 'both'):
        kNN_STD_means.append(kNearestNeighbor(
            features_data, labels_data, knnmodel))
        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        kNN_PCA_means.append(kNearestNeighbor(
            features_df_PCA, labels_data, knnmodel))

    else:
        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('kNearestNeighbors model with PCA')
        kNN_PCA_means.append(kNearestNeighbor(
            features_df_PCA, labels_data, knnmodel))

    # print()


def kNearestNeighborSimulation(knnmodel, linPCA, training_dataframe, pca_option):

    if(pca_option == 'both'):

        number = 20
        start = time.time()
        print('Simulating kNearestNeighbors models...')
        for i in range(0, number):
            #print('kNearestNeighbors simulation number', i, 'finished')
            mainkNearestNeighborImplementation(
                knnmodel, linPCA, training_dataframe, 'both')

        end = time.time()
        print('kNearestNeighbor Simulation time:', end - start)

        m = None

        if np.mean(kNN_STD_means) > np.mean(kNN_PCA_means):
            m = 'STANDARD MODEL'
        else:
            m = 'PCA TRANSFORMED MODEL'

        count = 0

        for i in range(0, number):
            if(kNN_PCA_means[i] > kNN_STD_means[i]):
                count = count + 1

        print()
        print('number of times pca transformed kNearestNeighbors model had greater accuracy than standard kNearestNeighbor model: ',
              count, 'out of ', number)
        print('kNearestNeighbors model variance in accuracies for standard model: ',
              np.var(kNN_STD_means))
        print('kNearestNeighbors model variance in accuracies for pca transform model: ',
              np.var(kNN_PCA_means))
        print('standard kNearestNeighbors model accuracy: ',
              np.mean(kNN_STD_means))
        print('pca transformed kNearestNeighbors model accuracy: ',
              np.mean(kNN_PCA_means))
        print('maximum knearestneighbors accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
            np.mean(kNN_PCA_means), np.mean(kNN_STD_means)), '%')
        print()
    elif(pca_option == 'yes'):
        number = 20
        print('Simulating kNearestNeighbors PCA model...')
        start = time.time()
        for i in range(0, number):
            #print('kNearestNeighbors simulation number', i, 'finished')
            mainkNearestNeighborImplementation(
                knnmodel, linPCA, training_dataframe, 'yes')

        end = time.time()
        print('kNearestNeighbor Simulation time:', end - start)

        print()
        print('kNearestNeighbors variance in accuracies for pca transform model: ',
              np.var(kNN_PCA_means))
        print('pca transformed kNearestNeighbors model accuracy on 10fold cross-val test data: ',
              np.mean(kNN_PCA_means), '%')
        print()
    else:
        number = 20
        print('Simulating kNearestNeighbors standard model...')
        start = time.time()
        for i in range(0, number):
            #print('kNearestNeighbors simulation number', i, 'finished')
            mainkNearestNeighborImplementation(
                knnmodel, linPCA, training_dataframe, 'no')

        end = time.time()
        print('kNearestNeighbor Simulation time:', end - start)

        print()
        print('standard kNearestNeighbors variance in accuracies: ',
              np.var(kNN_STD_means))
        print('standard kNearestNeighbors model accuracy on 10fold cross-val test data: ',
              np.mean(kNN_STD_means))
        print()


"""
    number = 20
    print('Simulating kNearestNeighbors model...')
    for i in range(0, number):
        #print('kNearestNeighbors simulation number', i, 'finished')
        mainkNearestNeighborImplementation(
            knnmodel, linPCA, training_dataframe)

    m = None

    if np.mean(kNN_STD_means) > np.mean(kNN_PCA_means):
        m = 'STANDARD MODEL'
    else:
        m = 'PCA TRANSFORMED MODEL'

    count = 0

    for i in range(0, number):
        if(kNN_PCA_means[i] > kNN_STD_means[i]):
            count = count + 1

    print()
    print('number of times kNearestNeighbors pca transformed model had greater accuracy than standard model: ',
          count, 'out of ', number)
    print('kNearestNeighbors variance in accuracies for standard model: ',
          np.var(kNN_STD_means))
    print('kNearestNeighbors variance in accuracies for pca transform model: ',
          np.var(kNN_PCA_means))
    print('kNearestNeighbors standard model accuracy: ', np.mean(kNN_STD_means))
    print('kNearestNeighbors pca transformed model accuracy: ',
          np.mean(kNN_PCA_means))
    print('MAXIMUM KNEARESTNEIGHBORS ACCURACY ATTAINED BY', m, "WITH AN ACCURACY OF", max(
        np.mean(kNN_PCA_means), np.mean(kNN_STD_means)), '%')
    print()
"""

knn = KNeighborsClassifier(n_neighbors=5)

"""
start = time.time()
mainkNearestNeighborImplementation(
    knn, processing.linear_pca, processing.overall_training_data, 'yes')
end = time.time()
print('Time of mainkNearestNeighborImplementation', end - start)
"""

# kNearestNeighborSimulation(knn, processing.linear_pca,
#                           processing.overall_training_data, option)
# kNearestNeighborSimulation(knn2, processing.linear_pca,
#                           processing.overall_training_data)
