import processing

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn import preprocessing
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



option = str(sys.argv[1])


DT_final_predictions = []


def DecisionTreeTest(pca_option):

    import DT

    DT.DecisionTreeSimulation(
        DT.dt, processing.linear_pca, processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global DT_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)
        final_predictions = DT.dt.predict(transformed_FV)
        DT_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('DECISION TREE MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('DECISION TREE MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('DECISION TREE MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()
        return accuracy, precision, recall

    else:

        final_predictions = DT.dt.predict(FV_features_data)
        DT_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('DECISION TREE MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('DECISION TREE MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('DECISION TREE MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


RF_final_predictions = []


def RandomForestTest(pca_option):

    import RF

    RF.RandomForestSimulation(
        RF.rf, processing.linear_pca, processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global RF_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)

        final_predictions = RF.rf.predict(transformed_FV)
        RF_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('RANDOM FOREST MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('RANDOM FOREST MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('RANDOM FOREST MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall
    else:

        final_predictions = RF.rf.predict(FV_features_data)
        RF_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('RANDOM FOREST MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('RANDOM FOREST MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('RANDOM FOREST MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


SVM_GAUS_final_predictions = []
SVM_POLY_final_predictions = []


def GaussianSVMTest(pca_option):

    import SVM

    SVM.SVMSimulation(SVM.svm_gaussian, processing.linear_pca,
                      processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global SVM_GAUS_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)

        final_predictions = SVM.svm_gaussian.predict(transformed_FV)
        SVM_GAUS_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('GAUSSIAN SVM MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('GAUSSIAN SVM MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('GAUSSIAN SVM MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall

    else:

        final_predictions = SVM.svm_gaussian.predict(FV_features_data)
        SVM_GAUS_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('GAUSSIAN SVM MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('GAUSSIAN SVM MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('GAUSSIAN SVM MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


def PolynomialSVMTest(pca_option):

    import SVM

    SVM.SVMSimulation(SVM.svm_poly, processing.linear_pca,
                      processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global SVM_POLY_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)

        final_predictions = SVM.svm_poly.predict(transformed_FV)
        SVM_GAUS_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('POLYNOMIAL SVM MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('POLYNOMIAL SVM MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('POLYNOMIAL SVM MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall

    else:

        final_predictions = SVM.svm_poly.predict(FV_features_data)
        SVM_GAUS_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('POLYNOMIAL SVM MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('POLYNOMIAL SVM MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('POLYNOMIAL SVM MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


kNN_final_predictions = []


def kNearestNeighborTest(pca_option):
    import kNN

    kNN.kNearestNeighborSimulation(
        kNN.knn, processing.linear_pca, processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global kNN_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)

        final_predictions = kNN.knn.predict(transformed_FV)
        kNN_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall

    else:

        final_predictions = kNN.knn.predict(FV_features_data)
        kNN_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('kNEARESTNEIGHBOR MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


NN_final_predictions = None


def NeuralNetworkTest(pca_option):

    import NN

    NN.NeuralNetworkSimulation(
        NN.nn, processing.linear_pca, processing.overall_training_data, pca_option)

    processing.final_validation = np.array(processing.final_validation)

    FV_features = []
    FV_labels = []

    FV_features, FV_labels = processing.createFeatures_Labels(
        processing.final_validation)

    FV_features_data = None
    FV_labels_data = None

    FV_features_data, FV_labels_data = processing.convertToDataFrame(
        FV_features, FV_labels, processing.column_titles)

    global NN_final_predictions
    if(pca_option == 'yes' or pca_option == 'both'):

        transformed_FV = processing.linear_pca.transform(FV_features_data)

        final_predictions = NN.nn.predict(transformed_FV)
        NN_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('NEURAL NETWORK MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('NEURAL NETWORK MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('NEURAL NETWORK MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall

    else:

        final_predictions = NN.nn.predict(FV_features_data)
        NN_final_predictions = final_predictions

        accuracy = metrics.accuracy_score(final_predictions, FV_labels)
        precision = metrics.precision_score(
            FV_labels, final_predictions, average='micro')
        recall = metrics.recall_score(
            FV_labels, final_predictions, average='micro')

        print('NEURAL NETWORK MODEL FINAL TEST DATA ACCURACY: ', 100 * accuracy)
        print('NEURAL NETWORK MODEL FINAL TEST DATA PRECISION: ', 100 * precision)
        print('NEURAL NETWORK MODEL FINAL TEST DATA RECALL: ', 100 * recall)
        print()

        return accuracy, precision, recall


def performanceMetrics(model_final_predictions, final_labels):
    accuracy = metrics.accuracy_score(model_final_predictions, final_labels)
    precision = metrics.precision_score(
        final_labels, model_final_predictions, average='micro')
    recall = metrics.recall_score(
        model_final_predictions, final_labels, average='micro')

    return accuracy, precision, recall


"""
dt_result = DecisionTreeTest(DT.dt, transformed_FV)
rf_result = RandomForestTest(RF.rf, transformed_FV)
svm_result = SVMTest(SVM.svm_gaussian, transformed_FV)
svm_result = SVMTest(SVM.svm_poly, transformed_FV)
knn_result = kNearestNeighborTest(kNN.knn, transformed_FV)
nn_result = NeuralNetworkTest(NN.nn, transformed_FV)
"""
option = str(sys.argv[1])

if(option == 'yes' or option == 'no' or option == 'both'):

    print()
    print()
    print('----FINAL PERFORMANCE METRICS----')
    print()
    print()


    kNN_accuracy, kNN_precision, kNN_recall = kNearestNeighborTest(option)

    DT_accuracy, DT_precision, DT_recall = DecisionTreeTest(option)

    RF_accuracy, RF_precision, RF_recall = RandomForestTest(option)

    NN_accuracy, NN_precision, NN_recall = NeuralNetworkTest(option)

    SVM_GAUS_accuracy, SVM_GAUS_precision, SVM_GAUS_recall = GaussianSVMTest(
        option)
    SVM_POLY_accuracy, SVM_POLY_precision, SVM_POLY_recall = PolynomialSVMTest(
        option)


    def findMaxMetric(metric_list):

        m_string = None
        m_int = max(metric_list)

        if(m_int == metric_list[0]):
            m_string = 'kNEARESTNEIGHBOR'
        elif(m_int == metric_list[1]):
            m_string = 'DECISION TREE'
        elif(m_int == metric_list[2]):
            m_string = 'RANDOM FOREST'
        elif(m_int == metric_list[3]):
            m_string = 'GAUSSIAN SVM'
        elif(m_int == metric_list[4]):
            m_string = 'POLYNOMIAL SVM'
        elif(m_int == metric_list[5]):
            m_string = 'NEURAL NETWORK'

        return m_int, m_string


    accuracy_list = [kNN_accuracy, DT_accuracy, RF_accuracy,
                     SVM_GAUS_accuracy, SVM_POLY_accuracy, NN_accuracy]
    precision_list = [kNN_accuracy, DT_accuracy, RF_accuracy,
                      SVM_GAUS_accuracy, SVM_POLY_accuracy, NN_accuracy]
    recall_list = [kNN_accuracy, DT_accuracy, RF_accuracy,
                   SVM_GAUS_accuracy, SVM_POLY_accuracy, NN_accuracy]


    max_accuracy, max_accuracy_string = findMaxMetric(accuracy_list)
    max_precision, max_precision_string = findMaxMetric(precision_list)
    max_recall, max_recall_string = findMaxMetric(recall_list)

    print('MAXIMUM ACCURACY ATTAINED BY', max_accuracy_string,
          'WITH ACCURACY OF', 100 * max_accuracy, '%')
    print('MAXIMUM PRECISION ATTAINED BY', max_precision_string,
          'WITH PRECISION OF', 100 * max_precision, '%')
    print('MAXIMUM RECALL ATTAINED BY', max_recall_string,
          'WITH RECALL OF', 100 * max_recall, '%')
else:

    print('Please input valid arguments: python3 FinalValidationTest.py <pca_optioin>')
    print("The options for <pca_option> should be 'yes', 'no', or 'both'")
    print("The 'yes' option trains the models on a PCA reduced dataset")
    print("The 'no' option trains the models on the standard non-PCA reduced dataset")
    print("The 'both' option trains the models on both the PCA reduced dataset and the standard non-PCA reduced dataset")
