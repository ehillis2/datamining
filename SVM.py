import processing

from sklearn.svm import SVC
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


def SVMModelScore(model, trainingdata, traininglabels, testingdata, testinglabels):
    model.fit(trainingdata, traininglabels.values.ravel())  # .values.ravel()
    return model.score(testingdata, testinglabels)

def SupportVectorMachine(data_features, data_labels, svmmodel):

    folds = StratifiedKFold(n_splits=10)

    scores = []

    for train_index, test_index in folds.split(data_features, data_labels):
        X_train, X_test, y_train, y_test = data_features.iloc[train_index], data_features.iloc[test_index], \
            data_labels.iloc[train_index], data_labels.iloc[test_index]
        scores.append(100 * SVMModelScore(svmmodel,
                                          X_train, y_train, X_test, y_test))

    #print('these are the scores: ', scores)
    #print('mean:', np.mean(scores))
    return np.mean(scores)


SVM_STD_means = []
SVM_PCA_means = []


def mainSVMImplementation(svmmodel, linPCA, training_dataframe, pca_option):

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

        #print('SVM model without PCA')
        SVM_STD_means.append(SupportVectorMachine(
            features_data, labels_data, svmmodel))

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('SVM model with PCA')
        SVM_PCA_means.append(SupportVectorMachine(
            features_df_PCA, labels_data, svmmodel))
    elif(pca_option == 'yes'):

        features_data_PCA = processing.linearPCAReduction(
            features_data, linPCA)
        features_df_PCA = pd.DataFrame(
            features_data_PCA, columns=column_titles[1:(features_data_PCA.shape[1] + 1)])

        #print('SVM model with PCA')
        SVM_PCA_means.append(SupportVectorMachine(
            features_df_PCA, labels_data, svmmodel))

    else:

        SVM_STD_means.append(SupportVectorMachine(
            features_data, labels_data, svmmodel))


def SVMSimulation(svmmodel, linPCA, training_dataframe, pca_option):

    if(svmmodel.kernel == 'rbf'):

        if(pca_option == 'both'):

            number = 20
            print('Simulating Gaussian SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Gaussian SVM Simulation time:', end - start)

            m = None

            if np.mean(SVM_STD_means) > np.mean(SVM_PCA_means):
                m = 'STANDARD MODEL'
            else:
                m = 'PCA TRANSFORMED MODEL'

            count = 0

            for i in range(0, number):
                if(SVM_PCA_means[i] > SVM_STD_means[i]):
                    count = count + 1

            print()
            print('number of times GAUSSIAN SVM pca transformed model had greater accuracy than standard model: ',
                  count, 'out of ', number)
            print('gaussian svm variance in accuracies for standard model: ',
                  np.var(SVM_STD_means))
            print('gaussian svm variance in accuracies for pca transform model: ',
                  np.var(SVM_PCA_means))
            print('gaussian svm standard model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_STD_means))
            print('gaussian svm pca transformed model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_PCA_means))
            print('maximum gaussian svm accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
                np.mean(SVM_PCA_means), np.mean(SVM_STD_means)), '%')
            print()

        elif(pca_option == 'yes'):

            number = 20
            print('Simulating PCA transformed Gaussian SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Gaussian SVM Simulation time:', end - start)

            print()
            print('gaussian svm variance in accuracies for pca transform model: ', np.var(
                SVM_PCA_means))
            print('gaussian svm pca transformed model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_PCA_means))
            print()
        else:

            number = 20
            print('Simulating standard Gaussian SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Gaussian SVM Simulation time:', end - start)

            print()
            print('pca transform gaussian svm variance model in accuracies: ', np.var(
                SVM_STD_means))
            print('pca transform gaussian svm model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_STD_means))
            print()

    else:

        if(pca_option == 'both'):

            number = 20
            print('Simulating Polynomial SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Polynomial SVM Simulation time:', end - start)

            m = None

            if np.mean(SVM_STD_means) > np.mean(SVM_PCA_means):
                m = 'STANDARD MODEL'
            else:
                m = 'PCA TRANSFORMED MODEL'

            count = 0

            for i in range(0, number):
                if(SVM_PCA_means[i] > SVM_STD_means[i]):
                    count = count + 1

            print()
            print('number of times Polynomial SVM pca transformed model had greater accuracy than standard model: ',
                  count, 'out of ', number)
            print('polynomial svm variance in accuracies for standard model: ',
                  np.var(SVM_STD_means))
            print('polynomial svm variance in accuracies for pca transform model: ',
                  np.var(SVM_PCA_means))
            print('polynomial svm standard model accuracy accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_STD_means))
            print('polynomial svm pca transformed model accuracy accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_PCA_means))
            print('maximum polynomial svm accuracy on 10fold cross-val test data attained by', m, "with an accuracy of: ", max(
                np.mean(SVM_PCA_means), np.mean(SVM_STD_means)), '%')
            print()

        elif(pca_option == 'yes'):

            number = 20
            print('Simulating PCA transformed Polynomial SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Polynomial SVM Simulation time:', end - start)

            print()
            print('polynomial svm variance in accuracies for pca transform model: ', np.var(
                SVM_PCA_means))
            print('polnomial svm pca transformed model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_PCA_means))
            print()
        else:

            number = 20
            print('Simulating standard polynomial SVM model... ')
            start = time.time()
            for i in range(0, number):
                #print('GAUSSIAN SVM simulation number', i, 'finished')
                mainSVMImplementation(
                    svmmodel, linPCA, training_dataframe, pca_option)
                # print()

            end = time.time()
            print('Polynomial SVM Simulation time:', end - start)

            print()
            print('pca transform polynomial svm variance model in accuracies: ', np.var(
                SVM_STD_means))
            print('pca transform polynomial svm model accuracy on 10fold cross-val test data: ',
                  np.mean(SVM_STD_means))
            print()


svm_gaussian = SVC(kernel='rbf')
svm_poly = SVC(kernel='poly')
"""
start = time.time()
mainSVMImplementation(
    svm_poly, processing.linear_pca, processing.overall_training_data)
end = time.time()
print('Time of mainPolynomialSVMImplementation', end - start)

start1 = time.time()
mainSVMImplementation(
    svm_gaussian, processing.linear_pca, processing.overall_training_data)
end1 = time.time()
print('Time of mainGaussianSVMImplementation', end1 - start1)

# SVMSimulation(svm_gaussian, processing.linear_pca,
#              processing.overall_training_data)
# SVMSimulation(svm_poly, processing.linear_pca,
#              processing.overall_training_data)
"""
