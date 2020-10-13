"""
k Nearest Neighbors Algorithm for Wisconsin Breast Cancer data

Ethan Hillis
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import operator
import sys
import math

"""
Randomly split the data for training and testing.
Parameters:
    dataset --> Complete data set
    split --> Percentage of data from dataset to be used for testing
    trainingdata --> Data to be used for training
    testingdata --> Data to be used for testing
Returns:
    Null
"""
def splitData(dataset, split, trainingdata, testingdata):

    for i in range(len(dataset)):
        if(random.random() < split):
            testingdata.extend([dataset.iloc[i]])
        else:
            trainingdata.extend([dataset.iloc[i]])

    testingdata = np.array(testingdata)
    trainingdata = np.array(trainingdata)

"""
Calculates the L2 norm (euclidean distance)between two observations
Parameters:
    obj1 --> first observation
    obj2 --> second observation
Returns:
    Distance between two observation
"""
def calculateDistance(obj1, obj2):

    object1 = np.array(obj1)
    object2 = np.array(obj2)
    resulting_row = np.array(object1 - object2)

    return np.linalg.norm(resulting_row)

"""
Finds the nearest neighbors of an observation in the data.
Parameters:
    observation --> observation you wish to find neighbors for
    data_set --> data the user wants to find neighbors from
    k --> number of nearest neighbors the user wants to find
Returns:
    Length k array containing the classification of nearest k neighbors
"""
def findNearestNeighbors(observation, dataset, k):
    # data set is entire data frame
    # test observation is the specific
    distances = []

    for i in range(len(dataset)):
        d = calculateDistance(observation[1:], dataset[i,1:])
        distances.append((dataset[i], d))

    distances.sort(key=operator.itemgetter(1))
    kdistances = distances[:k]
    neighbors = []

    for i in range(k):
        neighbors.append(distances[i][0][0])

    return neighbors

"""
Predict the classification of an observation based on its nearest
k neighbors.
Parameters:
    kneighbors --> array containing classification of nearest k neighbors
    k --> number of nearest neighbors the user wants to find
Returns:
    Predicted classification label of observation
"""
def predict(kneighbors, k):
    count = 0
    prediction = np.array([])

    for i in range(len(kneighbors)):
        prediction = kneighbors[i]
        if prediction == 2:
            count += 1
        else:
            count -= 1

    if count > 0:
        label = 2
    elif count < 0:
        label = 4
    else:
        label = random.choice([2,4])

    return label

"""
Calculate the accuracy of the model using the testing data
Parameters:
    testSet --> array of observations selected for testing
    predictions --> array of predictions for testing data
"""
def accuracy(testSet, predictions: list):
    predictions = np.array(predictions)
    correct_predictions = 0
    for i in range(len(testSet)):
        if testSet[i][0] == predictions[i]:
            correct_predictions += 1
    return (correct_predictions / float(len(testSet))) * 100.0

"""
Main function to run k Nearest Neighbor algorithm
Parameters:
    Null
Returns:
    Null
"""
def main():

    filename = sys.argv[-1]
    print('Randomly splitting data into training data and testing data...')

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.read_csv('/Users/ethanhillis/Desktop/Data Mining/BreastCancerData.csv')

    del df['id']
    trainingdata = []
    testingdata = []
    splitData(df, .20, trainingdata, testingdata)
    trainingdata = np.array(trainingdata)
    testingdata = np.array(testingdata)
    print('Number of observations in training data:', len(trainingdata))
    print('Number of observations in testing data:', len(testingdata))

    k = 5
    ma = []

    #Run algorithm and calculate accuracies for selected range of k's
    for x in range(1, int(math.sqrt(len(df)))):
        predictions = []
        for i in range(len(testingdata)):
            neighbors = findNearestNeighbors(testingdata[i], trainingdata, x)
            result = predict(neighbors, x)
            predictions.append(result)

        ma.append(round(accuracy(testingdata, predictions),5))

    #Print out accuracies for each k
    for i in range(len(ma)):
        print('Accuracy for k = ', 1+i, ':', ma[i])

    max_value = max(ma)
    max_index = 1 + ma.index(max_value)
    print('Maximum accuracy attained at k =', max_index, 'with accuracy of:',  max_value, '%')


"""
Run main function
Parameters:
    Null
Returns:
    Null
"""
main()
