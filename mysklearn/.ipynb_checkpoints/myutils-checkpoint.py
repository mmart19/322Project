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

def rank_by_bin_nums(data, num_bins):
    min_value = min(data)
    max_value = max(data)
    ranked_data = copy.deepcopy(data)
    
    value_range = max_value - min_value
    bin_range = value_range / num_bins
    
    cutoff_values = [0] * num_bins
    for i in range(len(cutoff_values)):
        cutoff_values[i] = round(min_value + bin_range * i, 3)
        
    for j in range(len(data)):
        if data[j] >= cutoff_values[0] and data[j] < cutoff_values[1]:
            ranked_data[j] = 1
        elif data[j] >= cutoff_values[1] and data[j] < cutoff_values[2]:
            ranked_data[j] = 2
        elif data[j] >= cutoff_values[2] and data[j] < cutoff_values[3]:
            ranked_data[j] = 3
        elif data[j] >= cutoff_values[3] and data[j] < cutoff_values[4]:
            ranked_data[j] = 4
        elif data[j] >= cutoff_values[4]:
            ranked_data[j] = 5
            
        
    return ranked_data

def get_column(table, col_index, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Index of identifier
        
        # Col array
        col = []
        # User doesn't want rows with missing values 
        if include_missing_values == False:
            for row in table: 
                # If the row isn't empty then append
                if row[col_index] != "NA":
                    col.append(row[col_index])
        # User wants rows with missing values
        else:
            # Append to the col
            for row in table:
                col.append(row[col_index])
        # return the col
        return col

def rank_by_bin_nums_2D(data, num_bins, col_index):
    data_list = get_column(data, col_index)
    min_value = min(data_list)
    max_value = max(data_list)
    
    value_range = max_value - min_value
    bin_range = value_range / num_bins
    
    cutoff_values = [0] * num_bins
    for i in range(len(cutoff_values)):
        cutoff_values[i] = round(min_value + bin_range * i, 3)
        
    for j in range(len(data)):
        if data[col_index][j] >= cutoff_values[0] and data[col_index][j] < cutoff_values[1]:
            data[col_index][j] = 1
        elif data[col_index][j] >= cutoff_values[1] and data[col_index][j] < cutoff_values[2]:
            data[col_index][j] = 2
        elif data[col_index][j] >= cutoff_values[2] and data[col_index][j] < cutoff_values[3]:
            data[col_index][j] = 3
        elif data[col_index][j] >= cutoff_values[3] and data[col_index][j] < cutoff_values[4]:
            data[col_index][j] = 4
        elif data[col_index][j] >= cutoff_values[4]:
            data[col_index][j] = 5
            
        
    return 0

def get_mode(data):
    
    values = []
    for item in data:
        if item not in values:
            values.append(item)
    
    counts = [0]* len(values)
    for i in range(len(values)):
        for k in range(len(data)):
            if data[k] == values[i]:
                counts[i] += 1
            
    max_index = 0 
    curr_max = 0
    for j in range(len(counts)):
        if counts[j] > curr_max:
            curr_max = counts[j]
            max_index = j
            
    return values[max_index]
    

def get_sum_stats(data):
    data_copy = copy.deepcopy(data)
    data_copy.sort()
    min_value = 0
    max_value = 0
    median = 0
    average = 0
    
    
    min_value = min(data_copy)
    max_value = max(data_copy)
    median = int(len(data_copy) / 2)
    
    if median % 2 == 0:
        median = data_copy[median / 2]
    else:
        median += .5
        median = int(median)
        median = data_copy[median]
        
    total = 0 
    for i in range(len(data_copy)):
        total += data_copy[i]
    average = round(total / len(data_copy), 3)
    
    mode = get_mode(data_copy)
    
    return min_value, max_value, median, average, mode
     
    


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

def get_average(seed_ppg):
    
    total = 0
    for i in range(len(seed_ppg)):
        total += seed_ppg[i]
    
    average = total / len(seed_ppg)
    
    return average

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
    
    label_counts = [0, 0]
    label_indices = [[], []]
    for i in range(len(y)):
        if y[i] == "No":
            label_counts[0] += 1
            label_indices[0].append(i)
        elif y[i] == "Yes":
            label_counts[1] += 1
            label_indices[1].append(i)
            
    
    label_percentages = [0, 0]
    for j in range(len(label_percentages)):
        label_percentages[j] = round(label_counts[j] / len(y), 2)
    
    label_splits = [0, 0]
    for index in range(len(label_splits)):
        label_splits[index] = round(label_counts[index] / n_splits, 0)
    
    for i in range(len(label_splits)):
        label_splits[i] = int(label_splits[i])
    
    
    #Grouping Folds
    x_fold = [] #* n_splits
    
    for i in range(n_splits):
        if i == 0:
            curr_no_max = label_splits[0] 
            curr_yes_max = label_splits[1] 
            j = 0
            curr_fold = []
            while j < curr_no_max:
                curr_fold.append(label_indices[0][j])
                j += 1
            j = 0
            while j < curr_yes_max:
                curr_fold.append(label_indices[1][j])
                j += 1
            x_fold.append(curr_fold)
        if i == 1:
            curr_no_max = label_splits[0] * 2
            curr_yes_max = label_splits[1] * 2
            j = label_splits[0] 
            curr_fold = []
            while j < curr_no_max:
                curr_fold.append(label_indices[0][j])
                j += 1
            j = label_splits[1] 
            while j < curr_yes_max:
                curr_fold.append(label_indices[1][j])
                j += 1
            x_fold.append(curr_fold)
        if i == 2:
            curr_no_max = label_splits[0] * 3
            curr_yes_max = label_splits[1] * 3
            j = label_splits[0] * 2
            curr_fold = []
            while j < curr_no_max -1:
                curr_fold.append(label_indices[0][j])
                j += 1
            j = label_splits[1] *2
            while j < curr_yes_max -1:
                curr_fold.append(label_indices[1][j])
                j += 1
            x_fold.append(curr_fold)
    return x_fold
    
def get_data_by_indice(X_train_indices, X_test_indices, table):
    X_train = []
    X_test = []
    y_test = []
    y_train = []
    
    train_indices = X_train_indices[0] + X_train_indices[1]
    test_indices = X_test_indices [0]
    
    total = len(train_indices)
    curr = 0
    while curr < total:
        for row in range(len(table)):
            if curr == total:
                break
            if train_indices[curr] == row:
                X_train.append(table[row])
                curr += 1
    
    total = len(test_indices)
    curr = 0
    while curr < total:
        for row in range(len(table)):
            if curr == total:
                break
            if test_indices[curr] == row:
                X_test.append(table[row])
                curr +=1
    
    print(len(X_train))
    print(len(X_test))
    
    for row in range(len(X_train)):
        y_train.append(X_train[row][-1])
        del X_train[row][-1]
        
    for row in range(len(X_test)):
        y_test.append(X_test[row][-1])
        del X_test[row][-1]
    
    return X_train, X_test, y_train, y_test
    
    
        
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

def get_sweet_overall_win_percentage(table):
    sweet_index = table.column_names.index("sweet-16?")
    percentage_index = table.column_names.index("Overall W-L%")
    win_percentage = []
    
    for row in range(len(table.data)):
        if (table.data[row][sweet_index] == "Yes"):
            win_percentage.append(table.data[row][percentage_index])
    
    return win_percentage

def get_sweet_conference_win_percentage(table):
    sweet_index = table.column_names.index("sweet-16?")
    percentage_index = table.column_names.index("Conference W-L%")
    win_percentage = []
    
    for row in range(len(table.data)):
        if (table.data[row][sweet_index] == "Yes"):
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



"""Function from class

"""
def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    test = [x for x in range(n)]
    for _ in range(n):
        rand_index = random.randrange(0,n)
        sample.append(table[rand_index])
        try:
            test.remove(rand_index)
        except ValueError:
            pass
    test_set = []
    for i in range(len(test)):
        test_set.append(table[test[i]])
    return sample, test_set


"""Function from class

"""
def compute_random_subset(values, num_values):
    shuffled = values[:] # shallow copy
    random.shuffle(shuffled)
    return shuffled[:num_values]
    
""" Function to get the frequency

"""
def get_frequency(pred):
    # Set items list
    items = []
    # Set frequency
    frequency = []
    # Traverse
    for val in pred:
        # Try 
        try:
            # Set index
            index = items.index(val)
            # Increment pos in frequency
            frequency[index] += 1
        # Wrong
        except ValueError:
            # Append to items
            items.append(val)
            # Append 1 
            frequency.append(1)
    # Return items and frequency
    return items, frequency
    