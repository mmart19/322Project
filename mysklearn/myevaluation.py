##############################################
# Programmer: Brian Steuber, Mateo Martinez 
# Class: CptS 322-01, Spring 2021
# Project
# 04/20/2021
# 
# Description: This program implements
# several functions that are at the 
# core of data science.
##############################################

import mysklearn.myutils as myutils
import copy
import numpy as np
import math
import random

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
       np.random.seed(random_state) 
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        # Traverse
        for index in range(len(X)):
            # Get random int
            rand = np.random.randint(0, (len(X)-1))
            # Creating tmp variables
            temp = X[rand]
            swap = X[rand-1]
            # Swapping values in X
            X[rand-1] = temp
            X[rand] = swap
    # Instance variable
    num_instances = len(X)
    # There is a float 
    if isinstance(test_size, float):
        # Generate the test size
        test_size = math.ceil(num_instances * test_size)
    # Get the split index
    split_index = num_instances - test_size   
    # Set X_train, X_test, y_train, y_test
    X_train = X[:split_index] 
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    # Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # Length of the 2d list
    n = len(X)
    # The list of training set indices for each fold
    X_train_folds = []
    # The list of testing set indices for each fold
    X_test_folds = []
    # The list of sample sizes
    sample_size = []
    # Number of folds
    folds = n % n_splits
    # Traverse
    for i in range(n_splits):
        # Append the sample size
        if i >= folds:
            sample_size.append(n // n_splits)
        # Append the sample size
        else:
            sample_size.append((n // n_splits) + 1)
    # Traverse
    for i in range(n_splits):
        # Create indices list
        indices = [j for j in range(len(X))]
        # Create the range_size 
        range_size = sample_size[i]
        # Create the start index 
        start_index = sum(sample_size[n] for n in range(i))
        # Create the test fold list
        test_fold = [k for k in range(start_index, start_index + range_size)]
        # Append test_fold list to X_test_folds
        X_test_folds.append(test_fold) 
        # Delete the indices from the start to start + range size
        del indices[start_index: start_index + range_size]
        # Append indices list to X_train_folds
        X_train_folds.append(indices)
    # Return X_train_folds, X_test_folds
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """    
    # Create a list of list with len of n_splits 
    #total_folds = [[] for _ in range(n_splits)]
    # The list of training set indices for each fold.
    #X_train_folds = [[] for _ in range(n_splits)]
    # The list of testing set indices for each fold.
    #X_test_folds = [[] for _ in range(n_splits)]
    # Call group to group the folds
    grouped_folds = myutils.group(X, y, n_splits)
    test_index = random.randrange(0, n_splits)
    X_train_folds = []
    X_test_folds = []
    
    for i in range(n_splits):
        if i == test_index:
            X_test_folds.append(grouped_folds[test_index])
        else:
            X_train_folds.append(grouped_folds[i])
    # Index variable
    #index = 0
    # Traverse
    #for row in grouped_folds:
        #for col in row:
            # Append the index sets
            #total_folds[index].append(col)
            #index = (index + 1) % n_splits
    # Reset index
    #index = 0
    # Traverse
    #for i in range(n_splits):
        #for j, row in enumerate(total_folds):
            # Not a match
            #if(i != j):
                #print("i: ",i)
                #print("j: ", j)
                # Traverse
                #for col in row:
                    # Append the index
                    #X_train_folds[index].append(col)
            # Match
            #else:
                #X_test_folds[index] = row
        # Increment the index
        #index += 1
    # Return X_train_folds, X_test_folds
    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Declare a label by label matrix of 0s 
    matrix = [[0 for x in labels] for y in labels]
    # Create a map {0: 0, 1: 1, 2: 2}
    a_map = {key: i for i, key in enumerate(labels)}
    # Create Confusion Matrix
    for p, a in zip(y_pred, y_true):
        matrix[a_map[a]][a_map[p]] += 1
    # Return Confusion Matrix
    return matrix
