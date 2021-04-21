##############################################
# Programmer: Brian Steuber, Mateo Martinez 
# Class: CptS 322-01, Spring 2021
# Project
# 04/20/2021
# 
# Description: This program provides
# several helper functions that are 
# used throughout this proj
##############################################

import copy
import random 
import math
#from tabulate import tabulate
from operator import itemgetter

# TODO: your reusable general-purpose functions here

""" Generates the Euclidean Distance

    Args: 
        v1: tuple of floats 
        v2: tuple of floats
        
    Returns:
        The distance betweeen the points
"""
def compute_euclidean_distance(v1, v2):
    #assert len(v1) == len(v2)
    #dist = -1
    #if(v1 == v2):
        #dist =  0
    #else:
        #dist = 1
    dist = (sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])) ** (1/2)
    return dist

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    
    cutoffs = []
    #cutoffs.append(min(values))
    count = 1
    for count in range(num_bins):
        cutoffs.append(min(values)+ bin_width * count)
    #cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs


""" Computes the slope and intercept
    
    Args:
        col1_data: list of x vals
        col2_data: list of y vals
    
    Returns:
        Slope and intercept
"""
def compute_slope(col1_data, col2_data):
    mean1 = sum(col1_data) / len(col1_data)
    mean2 = sum(col2_data) / len(col2_data)
    m = sum([(col1_data[i] - mean1) * (col2_data[i] - mean2) for i in range(len(col1_data))]) / sum([(col1_data[i] - mean1) ** 2 for i in range(len(col1_data))])
    b = mean2 - m * mean1
    return m, b

""" Normalizes the set of data
    
    Args: 
        x_train: list of lists - tuples
        y_train: list of lists - tuples
        
    Returns:
        Normalized data set

"""
def normalize(x_train, x_test):
    the_min_x1 = x_train[0][0]
    the_min_x2 = x_train[0][1] 
    for row in range(len(x_train)):
        if x_train[row][0] < the_min_x1:
            the_min_x1 = x_train[row][0]
        if x_train[row][1] < the_min_x2:
            the_min_x2 = x_train[row][1]

    for row in range(len(x_train)):
        x_train[row][0] = (x_train[row][0] - the_min_x1) 
        x_train[row][1] = (x_train[row][1] - the_min_x2)
    
    the_max_x1 = x_train[0][0]
    the_max_x2 = x_train[0][1]
    
    for row in range(len(x_train)):
        if x_train[row][0] > the_max_x1:
            the_max_x1 = x_train[row][0]
        if x_train[row][1] > the_max_x2:
            the_max_x2 = x_train[row][1] 
            
    for row in range(len(x_train)):
        x_train[row][0] /= the_max_x1
        x_train[row][1] /= the_max_x2

    x_test[0][0] = (x_test[0][0] - the_min_x1) / the_max_x1
    x_test[0][1] = (x_test[0][1] - the_min_x2) / the_max_x2
    
    return x_train, x_test

"""
    Function for itemgetter that is usually used with the operator library
"""
def itemgetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return obj[item]
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g



""" Helper function for stratified_kfold_cross_validation

    Args: 
        X: list of lists 
        y_train: list
        n_splits: float
        
    Returns:
        grouped_folds: a list of lists with the folds grouped
"""
def group(X, y, n_splits):
    # Represents the y_labels
    y_labels = []
    # List of the counts
    count_list = []
    # Traverse
    for label in y:
        # Not seen before
        if label not in y_labels:
            # Append the label
            y_labels.append(label)
            # Append 1 
            count_list.append(1)
        # Seen before
        elif label in y_labels:
            # Get the index
            index = y_labels.index(label)
            # Increment the instance
            count_list[index] += 1
    # Vals in X
    x_train = []
    # Y labels
    updated_y_labels = []
    # Traverse
    for x in range(len(y)):
        # Not seen before
        if y[x] not in updated_y_labels:
            # Append the index
            updated_y_labels.append(x)
            # Traverse
            for i in range(len(y_labels)):
                #Traverse
                for j in range(len(y)):
                    # Match
                    if y[j] == y_labels[i]:
                        # Append j to x_train
                        x_train.append(j)
        break
    # Empty list 
    grouped_folds = [[] for _ in range(n_splits)]
    # Element Count
    element_count = 0
    # Row Count
    row_count = -1
    # Traverse
    for i in range(len(x_train)):
        # Increase the row count if the mod of element count == 0
        if element_count % ((len(x_train) + 1) / 2) == 0:
            # Increment row count
            row_count += 1
        # Append to grouped_folds
        grouped_folds[row_count].append(x_train[element_count])
        # Increment element count
        element_count += 1
    # Return grouped_folds
    return grouped_folds
        
def get_vals(table, column_index, is_two_dim):
    vals = []
    final_vals = []
    for row in range(len(table)):
        vals.append(table[row][column_index])
    if is_two_dim == True:
        final_vals.append(vals)
        return final_vals
    return vals

def get_mpg_rating(mpg_list):
    mpg_ratings = []
    for row in range(len(mpg_list)):
        if mpg_list[row] <= 14.0:
            mpg_ratings.append(1)
        elif mpg_list[row] == 14.0:
            mpg_ratings.append(2)  
        elif mpg_list[row] > 14.0 and mpg_list[row] <= 16.0:
            mpg_ratings.append(3)
        elif mpg_list[row] > 16.0 and mpg_list[row] <= 19.0:
            mpg_ratings.append(4)
        elif mpg_list[row] > 19.0 and mpg_list[row] <= 23.0:
            mpg_ratings.append(5)
        elif mpg_list[row] > 23.0 and mpg_list[row] <= 26.0:
            mpg_ratings.append(6)
        elif mpg_list[row] > 26.0 and mpg_list[row] <= 30.0:
            mpg_ratings.append(7)
        elif mpg_list[row] > 30.0 and mpg_list[row] <= 36.0:
            mpg_ratings.append(8)
        elif mpg_list[row] > 36.0 and mpg_list[row] <= 44.0:
            mpg_ratings.append(9)
        elif mpg_list[row] > 44.0:
            mpg_ratings.append(10)
    return mpg_ratings

def get_mpg_rating2(mpg_list):
    mpg_ratings = []
    for row in range(len(mpg_list)):
        for col in range(len(mpg_list[0])): 
            if mpg_list[row][col] <= 14.0:
                mpg_ratings.append(1)
            elif mpg_list[row][col] == 14.0:
                mpg_ratings.append(2)  
            elif mpg_list[row][col] > 14.0 and mpg_list[row][col] <= 16.0:
                mpg_ratings.append(3)
            elif mpg_list[row][col] > 16.0 and mpg_list[row][col] <= 19.0:
                mpg_ratings.append(4)
            elif mpg_list[row][col] > 19.0 and mpg_list[row][col] <= 23.0:
                mpg_ratings.append(5)
            elif mpg_list[row][col] > 23.0 and mpg_list[row][col] <= 26.0:
                mpg_ratings.append(6)
            elif mpg_list[row][col] > 26.0 and mpg_list[row][col] <= 30.0:
                mpg_ratings.append(7)
            elif mpg_list[row][col] > 30.0 and mpg_list[row][col] <= 36.0:
                mpg_ratings.append(8)
            elif mpg_list[row][col] > 36.0 and mpg_list[row][col] <= 44.0:
                mpg_ratings.append(9)
            elif mpg_list[row][col] > 44.0:
                mpg_ratings.append(10)
    return mpg_ratings

def get_mpg_class(mpg):
    if(mpg >= 45):
        return 10
    elif(mpg >= 37 and mpg < 45):
        return 9
    elif(mpg >= 31 and mpg < 37):
        return 8
    elif(mpg >= 27 and mpg < 31):
        return 7
    elif(mpg >= 24 and mpg < 27):
        return 6
    elif(mpg >= 20 and mpg < 24):
        return 5
    elif(mpg >= 17 and mpg < 20):
        return 4
    elif(mpg >= 15 and mpg < 17):
        return 3
    elif(mpg >= 14 and mpg < 15):
        return 2
    else:
        return 1
    
def get_seed_counts(seed):
    """ Runs through the seed column determining counts for each seed

    Args: 
        seed (list of ints): list of ints
        
    Returns:
        seed_values (list of ints): All possible seed values
        seed_counts (list of ints): Counts for each seed
"""
    seed_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    seed_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    seed_copy = copy.deepcopy(seed)
    
    for value in range(len(seed_copy)):
        seed[value] = int(seed[value])
    
    for index in range(len(seed_copy)):
        if(seed_copy[index] == 1):
            seed_counts[0] += 1
        elif(seed_copy[index] == 2):
            seed_counts[1] += 1
        elif(seed_copy[index] == 3):
            seed_counts[2] += 1
        elif(seed_copy[index] == 4):
            seed_counts[3] += 1
        elif(seed_copy[index] == 5):
            seed_counts[4] += 1
        elif(seed_copy[index] == 6):
            seed_counts[5] += 1
        elif(seed_copy[index] == 7):
            seed_counts[6] += 1
        elif(seed_copy[index] == 8):
            seed_counts[7] += 1
        elif(seed_copy[index] == 9):
            seed_counts[8] += 1
        elif(seed_copy[index] == 10):
            seed_counts[9] += 1
        elif(seed_copy[index] == 11):
            seed_counts[10] += 1
        elif(seed_copy[index] == 12):
            seed_counts[11] += 1
        elif(seed_copy[index] == 13):
            seed_counts[12] += 1
        elif(seed_copy[index] == 14):
            seed_counts[13] += 1
        elif(seed_copy[index] == 15):
            seed_counts[14] += 1
        elif(seed_copy[index] == 16):
            seed_counts[15] += 1
            
    return seed_values, seed_counts

def get_elite_win_percentage(table):
    elite_index = table.column_names.index("elite8?")
    percentage_index = table.column_names.index("w-l%")
    print(elite_index)
    print(percentage_index)
    win_percentage = []
    
    for row in range(len(table.data)):
        if (table.data[row][elite_index] == "Yes"):
            win_percentage.append(table.data[row][percentage_index])
    
    return win_percentage

def convert_to_2d(self, column_name):
    index = self.index(column_name)
    arr = []
    
    for i in range(len(self)):
        curr_row = []
        curr_row.append(self[i][index])
        arr.append(curr_row)
    return arr

def print_step4(kfold_linearization_accuracy, kfold_knn_accuracy, strat_lin_accuracy, strat_knn_accuracy, tree1, tree2):
    print()
    print("=" * 80)
    print("Step 4: Predictive Accuracy")
    print("=" * 80)
    #print("10-Fold Cross Validation")
    #print("Linear Regression: accuracy = ", round(kfold_linearization_accuracy, 2), ", error rate = ", round(1-kfold_linearization_accuracy, 2), sep='')
    #kfold_knn_accuracy = 67/193
    #print("Naive Bayes: accuracy = ", round(kfold_knn_accuracy, 2), ", error rate = ", round(1-kfold_knn_accuracy, 2), sep='')
    #print()
    print("Stratified 10-fold Cross Validation")
    #strat_knn_accuracy = 74/193
    print("Linear Regression: accuracy = ", round(strat_lin_accuracy, 2), ", error rate = ", round(1-strat_lin_accuracy, 2), sep='')
    print("Naive Bayes: accuracy = ", round(strat_knn_accuracy, 2), ", error rate = ", round(1-strat_knn_accuracy, 2), sep='')
    print("Decision Tree: accurcay = ", round(tree1, 2), ", error rate = ", round(1-tree1, 2), sep='')


def print_step5(lin_matrix, knn_matrix, tree_matrix):
    print("=" * 80)
    print("Step 5: Confusion Matrices")
    print("=" * 80)
    print("Linear Regression (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(lin_matrix))
    print("\nNaive Bayes (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(knn_matrix))
    print("\nDecision Tree (Stratified 10-Fold Cross Validation Results):")
    print(tabulate(tree_matrix))
    
def getWeight(weight):
    if weight >= 3500:
        return 5
    elif weight >= 3000:
        return 4
    elif weight >= 2500:
        return 3
    elif weight >= 2000:
        return 2
    else:
        return 1
    
""" Helper for fit

    Args:
        X_train: List of Lists
        header: List
        
    Returns: attribute domains
"""
def get_attribute_domains(X_train, header):
    attribute_domains = {}
    for i, j in enumerate(header):
        attribute_domains[j] = []
        for x in X_train:
            if x[i] not in attribute_domains[j]:
                attribute_domains[j].append(x[i])
    for key, val in attribute_domains.items():
        attribute_domains[key] = sorted(val)
    return attribute_domains


""" Helper for calculate_entropy
    
    Args:
        instances: List of Lists
        index: Int

    Returns: Partitions for entropy
"""
def partition_for_entropy(instances, index):
    partitions = []
    unique = []
    for instance in instances:
        if instance[index] in unique:
            partition_idx = unique.index(instance[index])
            partitions[partition_idx].append(instance)
        else:
            unique.append(instance[index])
            partitions.append([instance])
    return partitions

""" Helper for select_attribute

    Args:
        instances: List of Lists
        available_attributes: List 
        
    Returns min Entropy
"""

def calculate_entropy(instances, available_attributes):
    attribute_entropies = []
    for attr in available_attributes:
        entropies = []
        denoms = []
        index = int(attr[-1])
        partitions = partition_for_entropy(instances, index)
        for partition in partitions:
            unique_classifiers = []
            classifiers_counts = []
            value_entropy = 0
            for instance in partition:
                if instance[-1] in unique_classifiers:
                    classifier_idx = unique_classifiers.index(instance[-1])
                    classifiers_counts[classifier_idx] += 1
                else:
                    unique_classifiers.append(instance[-1])
                    classifiers_counts.append(1)
            denom = len(partition)
            for count in classifiers_counts:
                if count == 0:
                    value_entropy = 0
                    break
                value_entropy -= count/denom * math.log(count/denom,2)
            entropies.append(value_entropy)
            denoms.append(denom/len(instances))
        total_entropy = 0
        for i in range(len(entropies)):
            total_entropy += entropies[i] * denoms[i]
        attribute_entropies.append(total_entropy)
    min_entropy = min(attribute_entropies)
    att_idx = attribute_entropies.index(min_entropy)
    return available_attributes[att_idx]

""" Function from class 

    Selects the attribute using Entropy 
"""

def select_attribute(instances, available_attributes):
    return calculate_entropy(instances, available_attributes)

""" Function from class

    Returns the partitions
""" 
def partition_instances(instances, split_attribute, attribute_domains, header):
    attribute_domain = attribute_domains[split_attribute] 
    attribute_index = header.index(split_attribute) 
    partitions = {} 
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions 

""" Function from class


    Returns: True or false depending on if all labels match the first label
"""

def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True 


""" Computes partition statistics 
        Args:
            current_instances: List
            index: Int
        Returns:
            final_stats: Vals to add to the tree
        """
def compute_partition_stats(current_instances, index):
    stats = {}
    for instance in current_instances:
        if instance[index] in stats:
            stats[instance[index]] += 1
        else:
            stats[instance[index]] = 1
    final_stats = []
    for key in stats:
        final_stats.append([key, stats[key]])
    return final_stats

""" Returns the tree 

    Args: 
        current_instances: List of Lists
        available_attributes: List 
        attribute_domains: Dictionary
        header: List
        
    Returns:
        The built up tree 
"""
def tdidt(current_instances, available_attributes, attribute_domains, header):
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    #print("splitting on:", split_attribute)
    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    #print("partitions:", partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        # TODO: append your leaf nodes to this list appropriately
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            # make a leaf node
            # look at class label and make an occurance of the most occuring class label????
            values_subtree.append(["Leaf", partition[0][-1], len(partition), len(current_instances)])
            
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            # handle clashes by creating a majority vote leaf node
            # Traverse y train?fpi
            partition_stats = compute_partition_stats(partition, -1)
            partition_stats.sort(key=lambda x:x[1])
            values_subtree.append(["Leaf", partition_stats[-1][0], len(partition), len(current_instances)])
            
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            # backtrack and replace the subtree w/a majority vote leaf node
            partition_stats = compute_partition_stats(current_instances, -1)
            partition_stats.sort(key=lambda x:x[1])
            leaf = ["Leaf", partition_stats[-1][0], len(partition), len(current_instances)]
            return leaf
        
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            values_subtree.append(subtree)
        tree.append(values_subtree)
    return tree


""" Helper function for predict
    
    Args:
        header: List
        tree: List of Lists...
        instance: List 

    Returns:
        The predicted class label
    
"""
def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    else: 
        return tree[1]

    
""" Helper for print_decision_rules

    Args:
        tree: List of Lists...
        stack: List of Tuples
        attribute_names: List
        class_name: String
  
    Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.
"""    
def print_helper(tree, stack, attribute_names, class_name):
    if(tree[0] == "Attribute"):
        for val in tree[2:]:
            add = [(tree[1], val[1])]
            print_helper(val[2], stack + add, attribute_names, class_name)
    elif(tree[0] == "Leaf"):
        print(get_rule(stack, tree[1], attribute_names, class_name))

""" Gets a single string for 1 rule

    Args:
        stack: List of Tuples
        value: String
        attribute_names: List
        class_name: String

    Returns a rule
"""
def get_rule(stack, value, attribute_names, class_name):
    if(attribute_names == None):
        name = lambda att: att
    else:
        name = lambda att: attribute_names[int(att[3:])]
    rule = "IF "
    for val in stack:
        rule += str(name(val[0])) + " == " + str(val[1]) + " "
        if(val == stack[-1]):
            rule += "THEN "
        else:
            rule += "AND "
    rule += str(class_name) + " = " + str(value)
    return rule
