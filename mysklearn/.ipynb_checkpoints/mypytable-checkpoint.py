##############################################
# Programmer: Brian Steuber, Mateo Martinez 
# Class: CptS 322-01, Spring 2021
# Project
# 04/20/2021
# 
# Description: This program implements
# several functions that are at the 
# core of data science. This MyPyTable is 
# used in our proj to do a series of 
# data set preperation tasks
##############################################


import mysklearn.myutils as myutils
import copy
import csv
from os.path import join
import random

#from numpy.lib.function_base import delete 
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    #def pretty_print(self):
         #"""Prints the table in a nicely formatted grid structure.
         #"""
         #print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        # Return the length 
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Index of identifier
        col_index = self.column_names.index(col_identifier)
        # Col array
        col = []
        # User doesn't want rows with missing values 
        if include_missing_values == False:
            for row in self.data: 
                # If the row isn't empty then append
                if row[col_index] != "NA":
                    col.append(row[col_index])
        # User wants rows with missing values
        else:
            # Append to the col
            for row in self.data:
                col.append(row[col_index])
        # return the col
        return col

    def get_sum(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Returns: 
            Sum of the col_identifier
        """
        index = self.column_names.index(col_identifier)
        sum = 0
        for row in range(len(self.data)):
            if not(isinstance(self.data[row][index], str)):
                sum += self.data[row][index]
        return sum


    def convert_to_integer(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Appends:
            Changes the data to an integer
        """
        index = self.column_names.index(col_identifier)
        for row in range(len(self.data)):
            try:
                the_int = int(self.data[row][index])
                self.data[row][index] = the_int
            except ValueError:
                pass

    def convert_to_string(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index

        Appends:
            Changes the data to an string
        """
        index = self.column_names.index(col_identifier)
        for row in range(len(self.data)):
            try:
                the_str = str(self.data[row][index])
                self.data[row][index] = the_str
            except ValueError:
                pass

    def mpg_rating(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
        
        Returns a list of ratings and the cooresponding counts
        """
        
        mpg_rating_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mpg_rating_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        index = self.column_names.index(col_identifier)
        for row in range(len(self.data)):
            if isinstance(self.data[row][index], str):
                pass
            else:
                if self.data[row][index] <= 14.0:
                    rating = 1
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] == 14.0:
                    rating = 2
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 14.0 and self.data[row][index] <= 16.0:
                    rating = 3
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 16.0 and self.data[row][index] <= 19.0:
                    rating = 4
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 19.0 and self.data[row][index] <= 23.0:
                    rating = 5
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 23.0 and self.data[row][index] <= 26.0:
                    rating = 6
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 26.0 and self.data[row][index] <= 30.0:
                    rating = 7
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 30.0 and self.data[row][index] <= 36.0:
                    rating = 8
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 36.0 and self.data[row][index] <= 44.0:
                    rating = 9
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
                elif self.data[row][index] > 44.0:
                    rating = 10
                    val_index = mpg_rating_list.index(rating)
                    mpg_rating_count[val_index] += 1
        return mpg_rating_list, mpg_rating_count
    
    def mpg_bins(self, arr, col_identifier):
        """
        Args:
            arr: list of data
            col_identifier(str or int): string for a column name or int
                for a column index
                
        Returns the rating count arr which contains the n unt given the len of arr
        """
        rating_count = [0] * len(arr)
        index = self.column_names.index(col_identifier)
        for row in range(len(self.data)):
            if self.data[row][index] >= arr[0] and self.data[row][index] < arr[1]:
                rating_count[0] += 1
            elif self.data[row][index] >= arr[1] and self.data[row][index] < arr[2]:
                rating_count[1] += 1
            elif self.data[row][index] >= arr[2] and self.data[row][index] < arr[3]:
                rating_count[2] += 1
            elif self.data[row][index] >= arr[3] and self.data[row][index] < arr[4]:
                rating_count[3] += 1
            elif self.data[row][index] >= arr[4]:
                rating_count[4] += 1
        return rating_count
    
    def movie_count(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
                
        Returns the movie count
        """
        
        index = self.column_names.index(col_identifier)
        count = 0
        for row in range(len(self.data)):
            if not(isinstance(self.data[row][index], str)):
                if self.data[row][index] == 1.0:
                    count += 1
        return count
    
    def remove_percent_sign(self, col_identifier):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
                
        Returns an arr of the col_identifier with the percent sign removed
        """
        ratings = []
        index = self.column_names.index(col_identifier)
        for row in range(len(self.data)):
            if len(self.data[row][index]) == 2:
                rating = self.data[row][index][0]
                the_float = float(rating)
                ratings.append(the_float)
            elif len(self.data[row][index]) == 3:
                rating = self.data[row][index][0:2]
                the_float = float(rating)
                ratings.append(the_float)
            elif len(self.data[row][index]) == 4:
                rating = self.data[row][index][0:3]
                the_float = float(rating)
                ratings.append(the_float)
        return ratings
                
    def calculate_genre_ratings(self, col_identifier, genre):
        """
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            genre(string): string for genre
                
        Returns the list of ratings for a paricular col
        """
        index = self.column_names.index(col_identifier)
        table = []
        for row in range(len(self.data)):
            if genre in self.data[row][index]:
                table.append(self.data[row])
        return table
    
    def generate_random_instances(self, n):
        table_len = len(self.data)
        random_instances = []
        for i in range(n):
            index = random.randint(0, table_len - 1)
            random_instances.append(self.data[index])
        return random_instances
    
    def print_instance(self, instance_title, random_vals, predicted, actual):
        print("===========================================")
        print(instance_title)
        print("===========================================")
        for row in range(len(random_vals)):
            print("instance: ", random_vals[row])
            print("class: ", predicted[row], " actual: ", actual[row])
            
    def print_step3(self, k, ratio, acc, err, acc2, err2):
        print("===========================================")
        print("Step 3: Predictive Accuracy")
        print("===========================================")
        print("Subsample (k=" + k + ", " + ratio + " Train/Test)")
        print("Linear Regression: accuracy = " + acc + ", error rate = " + err)
        print("Naive Bayes: accuracy = " + acc2 + ", error rate = " + err2)
        
    def print_step4(self, k, ratio, acc, err, acc2, err2, acc3, err3, acc4, err4):
        print("===========================================")
        print("Step 4: Predictive Accuracy")
        print("===========================================")
        print("10-Fold Cross Validation")
        print("Linear Regression: accuracy = " + acc + ", error rate = " + err)
        print("K Nearest Neighbors: accuracy = " + acc2 + ", error rate = " + err2)
        print()
        print("Stratified 10-Fold Cross Validation")
        print("Linear Regression: accuracy = " + acc3 + ", error rate = " + err3)
        print("K Nearest Neighbors: accuracy = " + acc4 + ", error rate = " + err4)
        
    def convert_to_2d(self, column_name):
        index = self.column_names.index(column_name)
        arr = []
        
        for i in range(len(self.data)):
            curr_row = []
            curr_row.append(self.data[i][index])
            arr.append(curr_row)
        return arr

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        # Traverse through rows
        for row in range(len(self.data)):
            # Traverse through cols
            for col in range(len(self.column_names)):
                # Try and convert
                try:
                    numeric_value = float(self.data[row][col])
                    self.data[row][col] = numeric_value
                # Otherwise throw an error
                except ValueError:
                    #print(self.data[row], " could not be converted to a numeric type")
                    pass
        # Return self object
        return self

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        # Arr representing dropped rows 
        droppedRows = []
        # Number of rows to drop
        num_rows_dropped = len(rows_to_drop)
        # Count variable
        count = 0

        if num_rows_dropped > 1:
            while count < num_rows_dropped:
                # Traverse through
                for row in range(len(self.data)):
                    if count < num_rows_dropped:
                        # Incerment counter
                        if self.data[row] == rows_to_drop[count]:
                            count += 1
                        # Append to dropped rows
                        else:
                            droppedRows.append(self.data[row])
                    else:
                        droppedRows.append(self.data[row])
        else:
            for row in range(len(self.data)):
                if count < 1:
                    if self.data[row] == rows_to_drop[count]:
                        count += 1
                    else:
                        droppedRows.append(self.data[row])
                else:
                    droppedRows.append(self.data[row])
                    
        # Copy droppedRows into self.data 
        self.data = copy.deepcopy(droppedRows)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # Open file
        infile = open(filename, "r", encoding='utf-8')
        # Make a reader with delimiter of ","
        reader = csv.reader(infile, delimiter = ",")
        # Read the first row
        firstRow = next(reader)
        # Read in first row to header
        for row in firstRow:
            self.column_names.append(row)
        # Read in rest of the data to table
        for row in reader:
            self.data.append(row)
        # Call convert to numeric
        self.convert_to_numeric()
        # Close the file
        infile.close()
        # TODO: finish this
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        out = open(filename, "w")
        writer = csv.writer(out)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        out.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        # Dups arr
        dups = []
        # Index arr
        indexes = []
        # Row arr
        rows = []
        # Traverse
        for i in range(len(key_column_names)):
            # Append to indexes
            indexes.append(self.column_names.index(key_column_names[i]))
        # Traverse
        for row in self.data:
            # Dup row arr
            dup_row = []
            # Traverse
            for i in indexes:
                # Add to dup row
                dup_row.append(row[i])
            # If there is a match
            if dup_row in rows:
                # Add to dups
                dups.append(row)
            # No match
            else:
                # Add to row arr
                rows.append(dup_row)
        # Return dups arr
        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        # Count var
        count = 0 
        # NewTable 
        newTable = []
        # Traverse the rows
        for row in range(len(self.data)):
            # Traverse the cols
            for col in range(len(self.column_names)):
                # There is a val
                if self.data[row][col] != "NA":
                    # Increment
                    count +=1
                # Append a nonempty row to the newTable
                else:
                    pass
                # There is no missing vals 
                if count == len(self.column_names):
                    newTable.append(self.data[row])
            # Reset counter
            count = 0
        # Clear data 
        self.data = []
        # Copy newTable
        self.data = copy.deepcopy(newTable)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # Index of col_name
        column_index = self.column_names.index(col_name)
        # Sum
        sum = 0
        # Number of valid entries
        non_empty_vals = 0

        # Traverse rows
        for row in range(len(self.data)):
            # Non empty row
            if self.data[row][column_index] != "NA":
                # Increment
                non_empty_vals += 1
                # Add to the sum
                sum += self.data[row][column_index]
        # Take the average
        average = sum / non_empty_vals
        # Traverse rows 
        for row in range(len(self.data)):
            # Empty
            if self.data[row][column_index] == "NA":
                # Replace with average
                self.data[row][column_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        # Table arr
        curr_table = []
        # Col index arr
        col_indexes = []
        # Col len
        col_len = len(col_names)
        # Index counter
        index = 0
        # Conditional
        while index < col_len: 
            # Traverse
            for column in range(len(self.column_names)):
                # Match
                if col_names[index] == self.column_names[column]:
                    # Add to the Index arr
                    col_indexes.append(column) 
            # Increment
            index += 1
        # Empty
        if not self.data:
            # Index
            index2 = 0 
            # Table arr
            curr_table = []
            # Conditional
            while index2 < len(col_indexes):
                # Stats arr
                curr_stats = []
                # Add to Stats arr
                curr_stats.append(self.column_names[col_indexes[index2]]) 
                # Add to Table
                curr_table.append(curr_stats)
                # Increment
                index2 += 1
            # Object            
            new_table = MyPyTable(curr_table, [])
            # Return
            return new_table
        # Not empty
        else:
            # Index
            index2 = 0 
            # Condtitional 
            while index2 < len(col_indexes):
                # Stats arr
                curr_stats = []
                # Add to stats arr
                curr_stats.append(self.column_names[col_indexes[index2]]) 
                # Set min
                min = self.data[0][col_indexes[index2]] 
                # Traverse
                for row in range(len(self.data)):
                    # Invalid
                    if(self.data[row][col_indexes[index2]] == "NA"):
                        pass
                    # Valid
                    else:
                        # Compare
                        if self.data[row][col_indexes[index2]] < min:
                            # Set Min
                            min = self.data[row][col_indexes[index2]]
                # Append min
                curr_stats.append(min)
                # Set max
                max  = self.data[0][col_indexes[index2]] 
                # Traverse
                for row in range(len(self.data)):
                    # Invalid
                    if(self.data[row][col_indexes[index2]] == "NA"):
                        pass
                    # Valid
                    else:
                        # Compare
                        if self.data[row][col_indexes[index2]] > max:
                            # Set Min
                            max = self.data[row][col_indexes[index2]]
                # Append max
                curr_stats.append(max)
                # Mid var
                mid = (min + max) / 2
                # Append mid
                curr_stats.append(mid)
                # Sum var
                sum = 0
                # counter
                count = 0
                # Traverse
                for row in range(len(self.data)):
                    # Invalid
                    if(self.data[row][col_indexes[index2]] == "NA"):
                        pass
                    # Valid
                    else:
                        # Add to sum
                        sum += self.data[row][col_indexes[index2]]
                        # Increment
                        count +=1
                # Create avg
                average = sum / count
                # Append avg
                curr_stats.append(average)
                # Sorted arr
                sorted = []
                # Traverse
                for row in range(len(self.data)):
                    # Invalid
                    if self.data[row][col_indexes[index2]] == "NA":
                        pass
                    # Valid
                    else:
                        # Add to sorted
                        sorted.append(self.data[row][col_indexes[index2]])
                # Sort the arr
                sorted.sort()
                # Even len arr
                if((len(sorted) % 2) == 0):
                    # Upper bound
                    upper_index = int(len(sorted) / 2)
                    # Median upper
                    median_upper = sorted[upper_index]
                    # Lower bound
                    lower_index = int(len(sorted) / 2 -1)
                    # Median lower
                    median_lower = sorted[lower_index]
                    # Median
                    median = (median_upper + median_lower) / 2
                    # Add median to stats
                    curr_stats.append(median)
                # Odd len arr
                else:
                    # Index
                    index = int((len(sorted) -1) / 2)
                    # Median
                    median = sorted[index]
                    # Add to stats
                    curr_stats.append(median)
                # Add stats to table
                curr_table.append(curr_stats)
                # Increment index
                index2 +=1
            # Names
            names = ["attributes", "min", "max", "mid", "avg", "median"]
            # Object
            new_table = MyPyTable(names, curr_table)
            # Return obj
            return new_table 


    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Col Names arr
        col_names = []
        # Data arr
        data = []
        # Copy of other_table.column_names
        col_copy = copy.deepcopy(other_table.column_names)
        # Traverse
        for title in key_column_names:
            # Remove the titles
            col_copy.remove(title)
        # Make a list of cols
        col_names = self.column_names + col_copy
        # Traverse
        for row in self.data:
            # Keys arr
            keys = []
            # Traverse
            for key in key_column_names:
                # Index
                index = self.column_names.index(key)
                # Val
                val = row[index]
                # Put in keys arr
                keys.append(val)
            # Traverse other table
            for row2 in other_table.data:
                # Another key arr
                more_keys = []
                # Traverse
                for key in key_column_names:
                    # Index
                    index = other_table.column_names.index(key)
                    # Val
                    val = row2[index]
                    # Put in key arr
                    more_keys.append(val)
                # Matches
                if keys == more_keys:
                    # Copy match
                    other_copy = copy.deepcopy(row2)
                    # Traverse
                    for val in more_keys:
                        # Remove dup
                        other_copy.remove(val)
                    # Row w/o dups
                    row_copy = row + other_copy
                    # Add to table
                    data.append(row_copy)
        # Make obj
        inner_join_table = MyPyTable(col_names, data)
        # Return the "3rd" table
        return inner_join_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # Col names arr
        col_names = []
        # Data arr 
        data = []

        # Copy of left table cols
        left_table_col_copy = copy.deepcopy(self.column_names)
        # Copy of right table cols
        right_table_col_copy = copy.deepcopy(other_table.column_names)
        # Copy of the left table data
        left_table_data_copy = copy.deepcopy(self.data)
        # Traverse the col_names
        for title in key_column_names:
            # Remove titles
            right_table_col_copy.remove(title)
            left_table_col_copy.remove(title)
        # Create a col list
        col_names = self.column_names + right_table_col_copy
        # Create index arr
        index_array = []
        # Traverse
        for row in left_table_data_copy:
            # Key arr
            self_keys = []
            # Bool for found
            found = False
            # Traverse and get vals
            for title in key_column_names:
                # Index
                index = self.column_names.index(title)
                # Val
                val = row[index]
                # Put in key arr
                self_keys.append(val)
            # Match var
            match_index = 0
            # Traverse other table
            for other_row in other_table.data:
                # Other key arr
                other_keys = []
                # Traverse and get vals
                for title in key_column_names:
                    # Index
                    index = other_table.column_names.index(title)
                    # Val
                    val = other_row[index]
                    # Put in other key arr
                    other_keys.append(val)
                # Match
                if self_keys == other_keys:
                    # Put in index arr
                    index_array.append(match_index)
                    # Set found to true 
                    found = True
                    # Other copy arr
                    other_copy = copy.deepcopy(other_row)
                    # Traverse
                    for key in other_keys:
                        # Remove dups
                        other_copy.remove(key)
                    # Add the row w/o dups
                    data_copy = row + other_copy
                    # Add to data arr
                    data.append(data_copy)
                # Increment match count
                match_index = match_index + 1
            # Not found
            if not found:
                # Outer join
                data_copy = row
                # Add NA
                for _ in right_table_col_copy:
                    data_copy.append("NA")
                # Add to Data
                data.append(data_copy)
        # Invalid Index arr
        invalid_indexes = []
        # Valid Index arr
        valid_indexes = []
        # Traverse
        for i in range(len(col_names)):
            # Not in other table
            if not col_names[i] in other_table.column_names:
                # Add to invalid arr
                invalid_indexes.append(i)
            # In other table
            else:
                # Add to valid arr
                valid_indexes.append(other_table.column_names.index(col_names[i]))
        # Other data arr
        other_data = copy.deepcopy(other_table.data)
        # Traverse
        for row in other_data:
            # Traverse
            for i in range(len(row)):
                # Delete rows we don't want
                if not i in valid_indexes:
                    del row[i]
        # Traverse
        for row in other_data:
            # Traverse 
            for i in invalid_indexes:
                # Add NA
                row.insert(i, "NA")

        # Traverse rows 
        for i in range(len(other_data)):
            # Valid 
            if not i in index_array:
                # Add to data arr
                data.append(other_data[i])
        
        # Create object
        outer_join_table = MyPyTable(col_names, data)
        # Return "3rd" table
        return outer_join_table