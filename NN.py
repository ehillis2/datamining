import processing

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import metrics
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


def NNmodelScore(model, trainingdata, traininglabels, testingdata, testinglabels):
    model.fit(trainingdata, traininglabels.values.ravel())
    return model.score(testingdata, testinglabels)

def NeuralNetwork(data_features, data_labels, nnmodel):

    folds = StratifiedKFold(n_splits=10)

    scores = []

    for train_index, test_index in folds.split(data_features, data_labels):
        X_train, X_test, y_train, y_test = data_features.iloc[train_index], data_features.iloc[test_index], \
            data_labels.iloc[train_index], data_labels.iloc[test_index]
        scores.append(100 * NNmodelScore(nnmodel,
                                         X_train, y_train, X_test, y_test))

    #print('these are the scores: ', scores)
    #print('mean:', np.mean(scores))
    return np.mean(scores)


NN_STD_means = []
NN_PCA_means = []

def mainNeuralNetworkImplementation(nnmodel, linPCA, training_dataframe, pca_option):

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
        NN_STD_means.append(NeuralNetwork(
            features_data, labels_data, nnmodel))
        # print()

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('Random Forest model with PCA')
        NN_PCA_means.append(NeuralNetwork(
            features_df_PCA, labels_data, nnmodel))
    elif(pca_option == 'yes'):

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('Random Forest model with PCA')
        NN_PCA_means.append(NeuralNetwork(
            features_df_PCA, labels_data, nnmodel))

    else:

        NN_STD_means.append(NeuralNetwork(
            features_data, labels_data, nnmodel))
    # print()


def NeuralNetworkSimulation(nnmodel, linPCA, training_dataframe, pca_option):

    if(pca_option == 'both'):

        start = time.time()

        number = 1
        print('Simulating neural network model...')
        for i in range(0, number):
            #print('neural network simulation number', i, 'finished')
            mainNeuralNetworkImplementation(
                nnmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Neural Network Simulation time:', end - start)

        m = None

        if np.mean(NN_STD_means) > np.mean(NN_PCA_means):
            m = 'STANDARD MODEL'
        else:
            m = 'PCA TRANSFORMED MODEL'

        count = 0

        for i in range(0, number):
            if(NN_PCA_means[i] > NN_STD_means[i]):
                count = count + 1

        print()
        print('number of times neural network pca transformed model had greater accuracy than random forest standard model: ',
              count, 'out of ', number)
        print('neural network variance in accuracy for standard model: ',
              np.var(NN_STD_means))
        print('neural network variance in accuracy for pca transform model: ',
              np.var(NN_PCA_means))
        print('neural network standard model accuracy: ', np.mean(NN_STD_means))
        print('neural network pca transformed model accuracy: ',
              np.mean(NN_PCA_means))
        print('maximum neural network accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
            np.mean(NN_PCA_means), np.mean(NN_STD_means)), '%')
        print()
    elif(pca_option == 'yes'):

        start = time.time()
        number = 1
        print('Simulating pca transformed neural network model...')
        for i in range(0, number):
            mainNeuralNetworkImplementation(
                nnmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Neural Network Simulation time:', end - start)
        print()
        print('pca transform neural network model variance in accuracy: ',
              np.var(NN_PCA_means))
        print('pca transformed neural network model accuracy on 10fold cross-val test data: ',
              np.mean(NN_PCA_means))
        print()

    else:
        number = 1
        start = time.time()
        print('Simulating standard neural network model...')
        for i in range(0, number):
            #print('neural network simulation number', i, 'finished')
            mainNeuralNetworkImplementation(
                nnmodel, linPCA, training_dataframe, pca_option)

        end = time.time()
        print('Neural Network Simulation time:', end - start)

        print()
        print('standard neural network model variance in accuracy: ',
              np.var(NN_STD_means))
        print('standard neural network model accuracy on 10fold cross-val test data: ',
              np.mean(NN_STD_means))
        print()


nn = MLPClassifier(random_state=1, solver='lbfgs')
nn_sgd = MLPClassifier(random_state=1, solver='sgd')


"""
start = time.time()
mainNeuralNetworkImplementation(
    nn, processing.linear_pca, processing.overall_training_data)
end = time.time()
print('Time of mainNeuralNetworkImplementation', end - start)

#NeuralNetworkSimulation(nn, processing.linear_pca,
#                        processing.overall_training_data)
"""
