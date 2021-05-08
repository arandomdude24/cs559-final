import numpy as np
import csv
import math

'''
Authors: Ishaan Patel & Dylan Lopez
CS 559 - Machine Learning Final Project
'''

def func():
    # open csv file and parse data
    f = open('Wynk.csv', 'r')
    csvreader = csv.reader(f)
    rows = []
    for row in csvreader:
        rows.append(row)

    rows = rows[1:] # cut out header row with titles
    
    # what features are we using
    # class is 11
    # feature columns: [0, 2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 17]
    classes = []
    data = []
    for row in rows:
        temp = [row[0]]
        temp.extend(row[2:5])
        temp.extend(row[7:11])
        temp.extend(row[15:18])
        data.append(temp)
        classes.append(row[11])
    data = np.array(data).astype(float)
    classes = np.array(classes).astype(int)
    print(classes)

    # repeat 10 times
    for loop in range(1):
        train_data = []
        test_data = []
        train_class = []
        test_class = []
        copy = data
        class_copy = classes

        # randomly split into test and train data
        for _ in range(len(int(data/2))):
            pass
        

        # check every test sample against every single train row to acquire
        # k nearest neighbours for each test sample




func()