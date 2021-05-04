##############################################
# Programmer: Brian Steuber
# Class: CptS 322-01, Spring 2021
# Programming Assignment #6
# 4/14/21
# 
# Description: This file provides
# several tests that test functions 
# used throughout this PA
##############################################

import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils
import random
from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyZeroRClassifier, MyRandomClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    # DataSet 1:
    np.random.seed(0)
    x = list(range(0, 100))
    y = [value * 2 + np.random.normal(0, 25) for value in x]
    test1_fit = MySimpleLinearRegressor()
    test1_fit.fit(x, y)
    stats_m, stats_b, stats_r, stats_r_p_val, stats_std_err = stats.linregress(x, y)
    assert np.isclose(test1_fit.slope, stats_m)
    assert np.isclose(test1_fit.intercept, stats_b)
    
    # DataSet2:
    np.random.seed(1)
    x1 = list(range(0, 100))
    y1 = [value * 2 + np.random.normal(0, 15) for value in x1]
    test2_fit = MySimpleLinearRegressor()
    test2_fit.fit(x1, y1)
    stats_m, stats_b, stats_r, stats_r_p_val, stats_std_err = stats.linregress(x1, y1)
    assert np.isclose(test2_fit.slope, stats_m)
    assert np.isclose(test2_fit.intercept, stats_b)

def test_simple_linear_regressor_predict():
    # DataSet 1:
    x_test1 = [[0], [1], [2], [3], [4], [5]]
    test_slope1 = 5
    test_intercept1 = 1
    test_predict1 = MySimpleLinearRegressor(test_slope1, test_intercept1)
    predicted_y_values1 = test_predict1.predict(x_test1)
    desk_answers1 = [1, 6, 11, 16, 21, 26]
    assert np.allclose(predicted_y_values1, desk_answers1)
    
    # DataSet2:
    x_test2 = [[0], [3], [6], [9], [12], [15]]
    test_slope2 = 8
    test_intercept2 = 3
    test_predict2 = MySimpleLinearRegressor(test_slope2, test_intercept2)
    predicted_y_values2 = test_predict2.predict(x_test2)
    desk_answers2 = [3, 27, 51, 75, 99, 123]
    assert np.allclose(predicted_y_values2, desk_answers2)

def test_kneighbors_classifier_kneighbors():
    # DataSet 1:
    x_train1 = [[7, 7], [7,4], [3,4], [1,4]]
    y_train1 = ["bad", "bad", "good", "good"]
    x_test1 = [[3, 7]]
    x_train1_final, x_test1_final = myutils.normalize(x_train1, x_test1)
    test_kneighbors1 = MyKNeighborsClassifier(4)
    test_kneighbors1.fit(x_train1_final, y_train1)
    distances1, indices1 = test_kneighbors1.kneighbors(x_test1_final)
    for i in range(len(distances1)):
        for j in range(len(distances1[0])):
            distances1[i][j] = round(distances1[i][j], 3)
    check_distance1 = [[.667, 1.00, 1.054, 1.202]]
    check_indices1 = [[0, 2, 3, 1]]
    assert np.allclose(distances1, check_distance1)
    assert np.allclose(indices1, check_indices1)
    
    # DataSet2:
    x_train2 = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    x_test2 = [[2, 3]]
    test_kneighbors2 = MyKNeighborsClassifier()
    test_kneighbors2.fit(x_train2, y_train2)
    distances2, indices2 = test_kneighbors2.kneighbors(x_test2)
    for i in range(len(distances2)):
        for j in range(len(distances2[0])):
            distances2[i][j] = round(distances2[i][j], 3)
    check_distance2 = [[1.414, 1.414, 2.00]]
    check_indices2 = [[0, 4, 6]]
    assert np.allclose(distances2, check_distance2)
    assert np.allclose(indices2, check_indices2)
    
    # DataSet3:
    x_train3 = [[0.8, 6.4], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6], [8.8, 9.8], [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], 
               [12.8, 1.1], [14.0, 19.9], [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], [19.0, 3.4], [19.6, 11.1]]
    y_train3 = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"]
    x_test3 = [[9.1, 11.0]]
    test_kneighbors3 = MyKNeighborsClassifier(20)
    test_kneighbors3.fit(x_train3, y_train3)
    distances3, indices3 = test_kneighbors3.kneighbors(x_test3)
    for i in range(len(distances3)):
        for j in range(len(distances3[0])):
            distances3[i][j] = round(distances3[i][j], 3)
    
    check_distance3 = [[0.608, 1.237, 2.202, 2.802, 2.915, 5.580, 6.807, 7.290, 7.871, 8.228, 8.645, 9.070, 9.122,
                    9.489, 9.981, 10.160, 10.500, 10.542, 10.569, 12.481]] 
    check_indices3 = [[6, 5, 7, 4, 8, 9, 14, 3, 2, 1, 15, 12, 13, 0, 17, 11, 19, 16, 10, 18]]
    check_predicted2 = ["yes"]
    assert np.allclose(distances3, check_distance3)
    assert np.allclose(indices3, check_indices3)

def test_kneighbors_classifier_predict():
    
    # DataSet1:
    x_train1 = [[7, 7], [7,4], [3,4], [1,4]]
    y_train1 = ["bad", "bad", "good", "good"]
    x_test1 = [[3, 7]]
    test_kneighbors1 = MyKNeighborsClassifier()
    test_kneighbors1.fit(x_train1, y_train1)
    y_predict1 = test_kneighbors1.predict(x_test1)
    check_predict1 = ["good"]
    #assert y_predict1 == check_predict1
    
    # DataSet2:
    x_train2 = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    x_test2 = [[2, 3]]
    test_kneighbors2 = MyKNeighborsClassifier()
    test_kneighbors2.fit(x_train2, y_train2)
    y_predict2 = test_kneighbors2.predict(x_test2)
    check_predict2 = ["yes"]
    #assert y_predict2 == check_predict2
    
    # DataSet3:
    x_train3 = [[0.8, 6.4], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6], [8.8, 9.8], [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], 
               [12.8, 1.1], [14.0, 19.9], [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9], [19.0, 3.4], [19.6, 11.1]]
    y_train3 = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"]
    x_test3 = [[9.1, 11.0]]
    test_kneighbors3 = MyKNeighborsClassifier()
    test_kneighbors3.fit(x_train3, y_train3)
    y_predict3 = test_kneighbors3.predict(x_test3)
    check_predict3 = ["+"]
    #assert y_predict3 == check_predict3
    
# Naive-bayes data
# First Dataset and Actuals
class_col_names = ["att1", "att2", "result"]
class_x_train = [
    [1, 5],
    [2, 6],
    [1, 5],
    [1, 5],
    [1, 6],
    [2, 6],
    [1, 5],
    [1, 6]
]
class_y_train = [
    "yes",
    "yes",
    "no",
    "no",
    "yes",
    "no",
    "yes",
    "yes"
]

# Class priors in order of "result = yes" then "result = no"
class_priors = {'no': 0.375, 'yes': 0.625}
# class_priors in list
"""
[.625, .375]
"""

# First row in class_posterior represent when "result = yes" and in order
# "att1 = 1", "att1 = 2", "att2 = 5", and "att2 = 6". Second row is same order
# except for when "result = no"
class_posteriors = {'no': [{1: 0.6666666666666666, 2: 0.3333333333333333}, {5: 0.6666666666666666, 6: 0.3333333333333333}], 'yes': [{1: 0.8, 2: 0.2}, {5: 0.4, 6: 0.6}]}
# class_posteriors in 2d list
"""
[
    [.8, .2, .4, .6],
    [.667, .333, .667, .333]
]
"""


# Unseen instanctes to run through Naive Bayes predict and
# desk answer predictions
class_test = [
    [1, 5]
    ]
class_actuals = ["yes"]

# Second Dataset and Actuals
iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
iphone_x_train = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
iphone_y_train = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes"
]

# iphone priors in order of "buys_iphone = yes" then "buys_iphone = no"
iphone_priors = {'no': 0.3333333333333333, 'yes': 0.6666666666666666}
# iphone_priors in list 
#[.333, .667]


# First row in iphone_posterior represent when "buys_iphone = yes" and in order
# "standing = 1", "standing = 2", "job_status = 1", "job_status = 2", "job_status = 3", 
# "credit_rating = fair" and "credit_rating = excellent". Second row is same order
# except for when "buys_iphone = no"
iphone_posteriors = {'no': [{1: 0.6, 2: 0.4}, {1: 0.2, 2: 0.4, 3: 0.4}, {'excellent': 0.6, 'fair': 0.4}], 'yes': [{1: 0.2, 2: 0.8}, {1: 0.3, 2: 0.4, 3: 0.3}, {'excellent': 0.3, 'fair': 0.7}]}
# iphone_posteriors in 2d list
"""
[
     [.6, .4, .2, .4, .4, .4, .6],
    [.2, .8, .3, .4, .3, .7, .3]
   
]
"""


# Unseen instanctes to run through Naive Bayes predict and
# desk answer predictions
iphone_test = [
    [2, 2, "fair"],
    [1, 1, "excellent"]
    ]
iphone_actuals = ["yes", "no"]

# Third Dataset and Actuals
train_col_names = ["day", "season", "wind", "rain", "class"]
train_x_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"], 
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
train_y_train = [
    "on time",
    "on time",
    "on time",
    "late", 
    "on time",
    "very late",
    "on time",
    "on time",
    "very late",
    "on time",
    "cancelled",
    "on time",
    "late",
    "on time",
    "very late",
    "on time",
    "on time",
    "on time",
    "on time",
    "on time"
]

# train priors in order of "class = on time", "class = late", "class = very late", then "class = cancelled"
train_priors = {'cancelled': 0.05, 'late': 0.1, 'on time': 0.7, 'very late': 0.15}
# train_priors in list
#[.7, .1, .15, .05]


# First row in iphone_posterior represent when "class = on time" and in order
# "day = weekday", "day = saturday", "day = sunday", "day = holiday", "season = spring", "season = summer", 
# "season = autumn", "season = winter", "wind = none", "wind = high", "wind = normal",
# "rain = none", "rain = slight", and "rain = heavy". Second and subsequent rows are
# same order except for when "class = late", "class = very late", and "class = cancelled"
train_posteriors = {
    'on time': [{'weekday': 0.6428571428571429, 'saturday': 0.14285714285714285, 'holiday': 0.14285714285714285, 'sunday': 0.07142857142857142}, {'spring': 0.2857142857142857, 'winter': 0.14285714285714285, 'summer': 0.42857142857142855, 'autumn': 0.14285714285714285}, {'none': 0.35714285714285715, 'high': 0.2857142857142857, 'normal': 0.35714285714285715}, {'none': 0.35714285714285715, 'slight': 0.5714285714285714, 'heavy': 0.07142857142857142}], 
    'late': [{'weekday': 0.5, 'saturday': 0.5, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 0.0, 'winter': 1.0, 'summer': 0.0, 'autumn': 0.0}, {'none': 0.0, 'high': 0.5, 'normal': 0.5}, {'none': 0.5, 'slight': 0.0, 'heavy': 0.5}], 
    'very late': [{'weekday': 1.0, 'saturday': 0.0, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 0.0, 'winter': 0.6666666666666666, 'summer': 0.0, 'autumn': 0.3333333333333333}, {'none': 0.0, 'high': 0.3333333333333333, 'normal': 0.6666666666666666}, {'none': 0.3333333333333333, 'slight': 0.0, 'heavy': 0.6666666666666666}], 
    'cancelled': [{'weekday': 0.0, 'saturday': 1.0, 'holiday': 0.0, 'sunday': 0.0}, {'spring': 1.0, 'winter': 0.0, 'summer': 0.0,'autumn': 0.0}, {'none': 0.0, 'high': 1.0, 'normal': 0.0}, {'none': 0.0, 'slight': 0.0, 'heavy': 1.0}]
}
# train_posteriors in 2d list
"""
[
    [.643, .143, .143, .071, .286, .143, .429, .143, .357, .357, .286, .357, .571, .071],
    [0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.333, 0.667, 0.0, 0.0, 0.667, 0.333, 0.0, 0.333, 0.667, 0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
]
"""


# Unseen instanctes to run through Naive Bayes predict and
# desk answer predictions
train_test = [
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "high", "heavy"],
    ["sunday", "summer", "normal", "sligh"]
    ]
train_actuals = ["very late", "on time", "on time"]

def test_naive_bayes_classifier_fit():
    # Test with 8 instance dataset from class
    test_fit = MyNaiveBayesClassifier()
    test_fit.fit(class_x_train, class_y_train)
    
    assert class_priors == test_fit.priors
    assert class_posteriors == test_fit.posteriors
    
    # Test Iphone dataset from Reading Quiz
    test_fit.fit(iphone_x_train, iphone_y_train)

    assert iphone_priors == test_fit.priors
    assert iphone_posteriors == test_fit.posteriors

    # Test train dataset from textbook
    test_fit.fit(train_x_train, train_y_train)
    
    assert train_priors == test_fit.priors
    assert train_posteriors == test_fit.posteriors
    
def test_naive_bayes_classifier_predict():
    # Setting up object to fit and predict class dataset
    test_predict = MyNaiveBayesClassifier()
    test_predict.fit(class_x_train, class_y_train)
    class_predicted = test_predict.predict(class_test)

    assert class_actuals == class_predicted

    # Setting up object to fit and predict iphone dataset
    test_predict.fit(iphone_x_train, iphone_y_train)
    iphone_predicted = test_predict.predict(iphone_test)
    
    assert iphone_actuals == iphone_predicted

    # Setting up object to fit and predict iphone dataset
    test_predict.fit(train_x_train, train_y_train)
    train_predicted = test_predict.predict(train_test)
    
    assert train_actuals == train_predicted
    
    # MyZeroRClassifier
    another = MyZeroRClassifier()
    another.fit(iphone_x_train, iphone_y_train)
    another_class = another.predict(iphone_test)
    #print(another_class)
    
    # MyRandomClassifier
    another2 = MyRandomClassifier()
    another2.fit(iphone_x_train, iphone_y_train)
    another_class2 = another2.predict(iphone_test)
    #print(another_class2)

# interview dataset
interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]


# bramer degrees dataset
degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

degrees_tree = \
    ['Attribute', 'att0', 
        ['Value', 'A', 
            ['Attribute', 'att4', 
                ['Value', 'A', 
                    ['Leaf', 'FIRST', 5, 14]
                ], 
                ['Value', 'B', 
                    ['Attribute', 'att3', 
                        ['Value', 'A', 
                            ['Attribute', 'att1', 
                                ['Value', 'A', 
                                    ['Leaf', 'FIRST', 1, 2]
                                ], 
                                ['Value', 'B', 
                                    ['Leaf', 'SECOND', 1, 2]
                                ]
                            ]
                        ], 
                        ['Value', 'B', 
                            ['Leaf', 'SECOND', 7, 9]
                        ]
                    ]
                ]
            ]
        ], 
        ['Value', 'B', 
            ['Leaf', 'SECOND', 12, 26]
        ]
    ]
    
def test_decision_tree_classifier_fit():
    # Interview DataSet
    
    # Create X_train and y_train
    X_train = []
    y_train = []
    # Append the header
    X_train.append(interview_header)
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)
    # Get y_train
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object 
    test_fit = MyDecisionTreeClassifier()
    # Call fit
    test_fit.fit(X_train, y_train)
    # Test 
    assert interview_tree == test_fit.tree
    
    # Degrees dataset
    
    # Create X_train2 and y_train2
    X_train2 = []
    y_train2 = []
    # Append the header
    X_train2.append(degrees_header)
    # Delete the classifier
    del X_train2[0][-1]
    # Get X_train
    for row in range(len(degrees_table)):
        tmp = []
        for col in range(len(degrees_table[0]) - 1):
            tmp.append(degrees_table[row][col])
        X_train2.append(tmp)
    # Get y_train
    for row in range(len(degrees_table)):
        y_train2.append(degrees_table[row][-1])
    # Create a MyDecisionTreeClassifier object
    test_fit2 = MyDecisionTreeClassifier()
    # Call fit
    test_fit2.fit(X_train2, y_train2)
    # Test 
    assert degrees_tree == test_fit2.tree
    

def test_decision_tree_classifier_predict():
    # Interview DataSet
    
    # Create X_train, y_train, X_test, and actuals
    X_train = []
    y_train = []
    X_test = [["Junior", "R", "yes", "no"], ["Junior", "Python", "no", "yes"], ["Senior", "Java", "no", "no", "False"]]
    actuals = ["True", "True", "False"]
    # Append the header
    X_train.append(interview_header)
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train2
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)
    # Get y_train2
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object
    test_predict = MyDecisionTreeClassifier()
    # Call fit
    test_predict.fit(X_train, y_train)
    # Call predict 
    predicted = test_predict.predict(X_test)
    # Test 
    assert predicted == actuals
    
    # Degrees dataset
    
    # Create X_train2, y_train2, X_test2, and actuals2
    X_train2 = []
    y_train2 = []
    X_test2 = [["A", "B", "A", "B", "B"], ["B", "A", "A", "B", "B"], ["A", "A", "B", "A", "A"]]
    actuals2 = ["SECOND", "SECOND", "FIRST"]
    # Append the header
    X_train2.append(degrees_header)
    # Delete the classifier
    del X_train2[0][-1]
    # Get X_train2
    for row in range(len(degrees_table)):
        tmp = []
        for col in range(len(degrees_table[0]) - 1):
            tmp.append(degrees_table[row][col])
        X_train2.append(tmp)
    # Get y_train2
    for row in range(len(degrees_table)):
        y_train2.append(degrees_table[row][-1])
    # Create a MyDecisionTreeClassifier object
    test_predict2 = MyDecisionTreeClassifier()
    # Call fit
    test_predict2.fit(X_train2, y_train2)
    # Call predict 
    predicted2 = test_predict2.predict(X_test2)
    # Test
    assert predicted2 == actuals2
    
tree_actual_1 = ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 2, 13]], ['Value', 'Mid', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 1,4]], ['Value', 'Python', ['Leaf','False', 0, 2]], ['Value', 'R', ['Leaf', 'False', 1, 4]]]], ['Value', 'Senior', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'True', 2, 7]], ['Value', 'Python', ['Leaf', 'False', 3, 7]], ['Value', 'R', ['Leaf', 'True', 2, 7]]]]]

tree_actual_2 = ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 0, 7]], ['Value', 'Mid', ['Leaf', 'True', 2, 13]], ['Value', 'Senior', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 0, 2]], ['Value', 'Python', ['Leaf', 'False', 1, 4]], ['Value', 'R', ['Leaf', 'True', 1, 4]]]]]


def test_MyRandomForestClassifier_fit():
    random.seed(1)
    # Interview DataSet
    
    # Create X_train and y_train
    X_train = []
    y_train = []
    # Append the header
    X_train.append(["level", "lang", "tweets", "phd", "interviewed_well"])
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)
    
    # Get y_train
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object 
    #print(X_train)
    test_fit = MyRandomForestClassifier(100, 2, 2)
    # Call fit
    
    test_fit.fit(X_train, y_train)
    # Test 
    #print("working")
    #print(test_fit.forest)
    
    assert(test_fit.forest[0]['atts']) == ['att0', 'att1']
    assert(test_fit.forest[0]['tree'].tree) == tree_actual_1
    
    assert(test_fit.forest[1]['atts']) == ['att0', 'att1']
    assert(test_fit.forest[1]['tree'].tree) == tree_actual_2
    
def test_MyRandomForestClassifier_predict():
    random.seed(1)
    # Interview DataSet
    
    # Create X_train and y_train
    X_train = []
    y_train = []
    X_test = [["Junior", "R", "yes", "no"], ["Junior", "Python", "no", "yes"], ["Senior", "Java", "no", "no", "False"]]
    # Append the header
    X_train.append(["level", "lang", "tweets", "phd", "interviewed_well"])
    # Delete the classifier
    del X_train[0][-1]
    # Get X_train
    for row in range(len(interview_table)):
        tmp = []
        for col in range(len(interview_table[0]) - 1):
            tmp.append(interview_table[row][col])
        X_train.append(tmp)
    
    # Get y_train
    for row in range(len(interview_table)):
        y_train.append(interview_table[row][-1])
    # Create a MyDecisionTreeClassifier object 
    #print(X_train)
    test_fit = MyRandomForestClassifier(100, 2, 2)
    # Call fit
    actual = ['True', 'True', 'True']
    test_fit.fit(X_train, y_train)
    predicted = test_fit.predict(X_test)
    assert predicted == actual
    
    
     