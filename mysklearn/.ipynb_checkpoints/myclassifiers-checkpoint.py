##############################################
# Programmer: Brian Steuber, Mateo Martinez 
# Class: CptS 322-01, Spring 2021
# Project
# 04/20/2021
# 
# Description: This program provides
# classes that are used throughout this proj
##############################################

import mysklearn.myutils as myutils
import copy
import random
from operator import itemgetter

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        self.slope, self.intercept = myutils.compute_slope(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        # The predicted target y values
        y_predicted = []
        # Traverse
        for list in X_test:
            for element in list:
                # Calculate y
                y = (self.slope * element) + self.intercept 
                # Append to y_predicted
                y_predicted.append(y)
        # Return y_predicted
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        # Make a copy of X_train
        distances_copy = copy.deepcopy(self.X_train)
        # New distances arr
        new_distances = []
        # Get distance, append to instance, append label, append index
        for i, instance in enumerate(distances_copy):
            instance.append(self.y_train[i])
            instance.append(i)
            dist = (myutils.compute_euclidean_distance(instance[:len(X_test[0])], X_test[0]))
            instance.append(dist)
        # Put instances in new_distances
        for instance in distances_copy:
            new_distances.append(instance)
        # Sort
        train_sorted = sorted(distances_copy, key=myutils.itemgetter(-1))
        # Set the top
        top = train_sorted[:self.n_neighbors]
        # 2D list of k nearest neighbor distances for each instance in X_test
        distances = []
        # 2D list of k nearest neighbor indices in X_train
        neighbor_indices = []
        # Tmp list
        tmp = []
        # Tmp2 list
        tmp2 = []
        # Traverse
        for i in range(self.n_neighbors):
            # Append the distance rounded to 3 decimal points 
            # I felt this was adequate enough 
            tmp.append(round(top[i][-1], 3))
            # Append the indice
            tmp2.append(top[i][3])
        # Append to distances
        distances.append(tmp)
        # Append to neighbor_indices
        neighbor_indices.append(tmp2)
        # Return distances, neighbor_indices
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # The predicted target y values
        y_predicted = []
        # Call kneighbors to set x1, x2
        x1,x2 = self.kneighbors(X_test)
        
        # Convert x2 to 1d arr 
        one_d_arr = []
        for i in range(self.n_neighbors):
            one_d_arr.append(x2[0][i])
        # Neighbors
        neighbors = {}
        # Find the class
        """
        for i in range(len(one_d_arr)):
            one_d_arr[i] = int(one_d_arr[i])
        """
        
        
        for idx in range(len(one_d_arr)):
            if self.y_train[idx] in neighbors:
                neighbors[self.y_train[idx]] += 1
            else:
                neighbors[self.y_train[idx]] = 1
        # Sort the neighbors
        sorted_neighbors = sorted(neighbors.items(), key=myutils.itemgetter(1), reverse=True)
        # Add class label
        y_predicted.append(sorted_neighbors[0][0])
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # Set x and y train
        self.X_train = X_train
        self.y_train = y_train
        # Set priors and posteriors to empty dictionaries
        self.priors = {}
        self.posteriors = {}
        # Create list
        arr = []
        # Traverse y_train and get counts of priors
        for val in y_train:
            # There is an instance
            if val in self.priors:
                # Increment
                self.priors[val] += 1
            # There is no instance
            else:
                # Set to 1
                self.priors[val] = 1    
        # Traverse X_train
        for row in X_train:
            # Traverse row 
            for i, j in enumerate(row):
                # Compare
                if i >= len(arr):
                    # Append empty dictionary
                    arr.append({})
                # There is not an instance
                if j not in arr[i]:
                    # Set to 0
                    arr[i][j] = 0
        # Traverse Priors
        for val in self.priors:
            # Copy
            self.posteriors[val] = copy.deepcopy(arr)
        # Traverse X/y_train
        for row, x in zip(X_train, y_train):
            # Traverse row 
            for i, j in enumerate(row):
                # Increment
                self.posteriors[x][i][j] += 1
        # Traverse posteriors
        for val in self.posteriors:
            # Traverse 
            for i, row in enumerate(self.posteriors[val]):
                # Traverse
                for val2 in row:
                    # Balance Posteriors
                    self.posteriors[val][i][val2] /= self.priors[val] 
        # Balance Priors
        for val in self.priors:
            self.priors[val] /= len(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Create list
        y_predicted = []
        # Traverse X_test
        for row in X_test:
            # Probability dictionary
            probs = {}
            # Traverse Posteriors
            for val in self.posteriors:
                # Set probs
                probs[val] = self.priors[val]
                # Traverse
                for i, j in enumerate(row):
                    # Try block
                    try:
                        # Multiply
                        probs[val] *= self.posteriors[val][i][j]
                    # Something went wrong
                    except:
                        pass
            # Max Vars
            max_string = ""
            max_val = -1
            # Traverse probs
            for val in probs:
                # Comparison
                if probs[val] > max_val:
                    # Set the max_string
                    max_string = val
                    # Set max_val
                    max_val = probs[val]
            # Append the key
            y_predicted.append(max_string)     
        # Return y_predicted
        return y_predicted
        
class MyZeroRClassifier:
    def __init__(self):
        """Initializer for MyZeroRClassifier.

        """
        self.X_train = None 
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Fits a MyZeroR classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Create list
        y_predicted = []
        # Traverse y_train
        for val in self.y_train:
            # There is no instance in y_predicted
            if val not in y_predicted:
                # Append Val
                y_predicted.append(val)
        # Create a count list initialized to 0s
        count = [0 for val in y_predicted]
        # Traverse y_train
        for val in self.y_train:
            # Traverse y_predicted
            for i, j in enumerate(y_predicted):
                # Match
                if val == j:
                    # Increment count
                    count[i] += 1
        # Max Var
        the_max = 0
        # Traverse count
        for i, val in enumerate(count):
            # New max
            if count[the_max] < count[i]:
                # Set the max
                the_max = i
        # Return y_predicted
        return [y_predicted[the_max] for val in X_test]
    
    
class MyRandomClassifier:
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None 
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Fits a MyZeroR classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Length of y_train
        the_len = len(self.y_train)
        # Random index
        rand_num = random.randint(0, the_len-1)
        # Set y_predicted
        y_predicted = [self.y_train[rand_num] for val in X_test]
        # Return y_predicted
        return y_predicted
    
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train, allowed_attributes = None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Set X_train and y_train
        self.X_train = X_train
        self.y_train = y_train
        # Set the header
        header = []
        # Delete header from X_train
        # Make a copy of the header
        if allowed_attributes is not None:
            header = allowed_attributes
        else:
            header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        del self.X_train[0] 
        available_attributes = header.copy()
        # Get the attribute domains
        attribute_domains = myutils.get_attribute_domains(self.X_train, header)
        # Create the train
        train = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
        # Call tdidt to set the tree
        self.tree = myutils.tdidt(train, available_attributes, attribute_domains, header)
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Create y_predicted
        y_predicted = []
        # Set the header
        header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        # Traverse the X_test
        for test in X_test:
            # Call predict on test and append that to y_predicted
            y_predicted.append(myutils.tdidt_predict(header, self.tree, test))
        # Return y_predicted
        return y_predicted
      
        
    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # Call print_helper function
        myutils.print_helper(self.tree, [], attribute_names, class_name)
       
        
    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    def __init__(self, N, M, F):
        """Initializer for MyRandomForestClassifier.
        
        """
        self.X_train = None 
        self.y_train = None
        self.forest = None
        self.N = N
        self.M = M
        self.F = F
        
    def fit(self, X_train, y_train):
        # Set X_train
        self.X_train = X_train
        # Set y_train
        self.y_train = y_train
        # Set header
        header = ['att' + str(i) for i in range(len(self.X_train[0]))]
        # Delete header
        del self.X_train[0]
        # Set train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # Create forest array
        forest = []
        # Traverse
        for _ in range(self.N):
            # Create tree dictionary
            tree = {}
            # Set attributes
            tree['atts'] = myutils.compute_random_subset(header[:-1], self.F)
            # Get train set and test set
            train_set, test_set = myutils.compute_bootstrapped_sample(train)
            # Set tree
            tree['tree'] = self.get_tree(train_set, tree['atts'] + [header[-1]])
            # Set accuracy
            tree['accuracy'] = self.compute_tree_accuracy(tree, test_set)
            # Append tree to forest
            forest.append(tree)
        # Sort
        sort = sorted(forest, key=itemgetter('accuracy'), reverse=True)
        # Set forest
        self.forest = sort[:self.M]
        
    def predict(self, X_test):
        # Create y_predicted list
        y_predicted = []
        # Predicted list
        predicted = []
        # Traverse forest
        for tree in self.forest:
            # Get the instance
            the_tree = tree['tree']
            # Call predict
            prediction = the_tree.predict(X_test)
            # Append to predicted
            predicted.append(prediction)
        # Traverse X_test
        for i in range(len(X_test)):
            # Create curr list
            curr = []
            # Traverse forest
            for j in range(len(self.forest)):
                # Append predicted[j][i] to curr
                curr.append(predicted[j][i])
            # Get items and frequency
            items, frequency = myutils.get_frequency(curr)
            # Get the best
            best = max(frequency)
            # Get the index of the best 
            the_index = frequency.index(best)
            # Append the prediction to y_prediction
            y_predicted.append(items[the_index])
        # Return y_predicted
        return y_predicted
    
    def get_tree(self, train, attr):
        # Create an instance of MyDecisionTreeClassifier 
        tree = MyDecisionTreeClassifier()
        # Set x 
        x = [row[:-1] for row in train]
        # Set y 
        y = [col[-1] for col in train]
        # Fit tree 
        tree.fit(x, y, attr)
        # Return tree
        return tree
    
    def compute_tree_accuracy(self, tree, test_set):
        # Set x 
        x = [row[:-1] for row in test_set]
        # Set Actual
        actual = [row[-1] for row in test_set]
        # Set predicted
        predicted = tree['tree'].predict(x)
        # Set total
        total = 0
        # Traverse 
        for i in range(len(actual)):
            # Match
            if actual[i] == predicted[i]:
                # Increment total
                total += 1
        # Set accuracy
        accuracy = total / len(actual)
        # Return accuracy
        return accuracy
    
    