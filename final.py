import numpy as np
import csv
import math
from scipy import stats

'''
Authors: Ishaan Patel & Dylan Lopez
CS 559 - Machine Learning Final Project
'''
# START: Ishaan Patel

# calculates the distance between 2 arrays of the same length
# returns a tuple of (distance, class of training sample)
def dist(test, train, train_class):
    dist = 0.0
    for i in range(11):
        dist += (test[i] - train[i]) ** 2
    return (math.sqrt(dist), train_class)

# distance formula but after applying FLD
def dist_FLD(test, train, train_class):
    dist = (test - train)**2
    return (math.sqrt(dist), train_class)

# computes the scatter matrix for a class input
def scatter_matrix(input, mean):
    scat = [[0 for i in range(len(mean))] for j in range(len(mean))]
    for sample in input:
        x = np.array(np.vstack(sample - mean))
        x_tp = np.transpose(x)
        mult = np.matmul(x, x_tp)
        scat += mult
    return np.array(scat)

# the main function, k represents how many nearest neighbours you want
# ASSUMPTIONS: K is a positive integer (i did not error handle the parameter!)
def func(k):
    # open csv file and parse data
    print("Opening and parsing file...")
    f = open('Wynk.csv', 'r')
    csvreader = csv.reader(f)
    rows = []
    for row in csvreader:
        rows.append(row)

    rows = rows[1:] # cut out header row with titles
    
    # what features we are using
    # class is column 11
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

    data = data[:-89]
    data = np.array(data).astype(float)
    classes = np.array(classes).astype(int)
    print("Finished parsing data...")

    pre_accuracy = []
    post_accuracy = []

    print("Beginning 10 trials...\n")
    # repeat 10 times
    for loop in range(10):
        train_data = []
        test_data = []
        train_class = []
        test_class = []
        copy = data
        class_copy = classes

        # randomly split into test and train data
        for j in range(int(len(data)/100)):
            random_row = np.random.randint(len(copy), size=1)
            test_data.append(copy[random_row].flatten())
            test_class.append(class_copy[random_row])
            class_copy = np.delete(class_copy, random_row, 0)
            copy = np.delete(copy, random_row, 0)

            random_row = np.random.randint(len(copy), size=1)
            train_data.append(copy[random_row].flatten())
            train_class.append(class_copy[random_row])
            class_copy = np.delete(class_copy, random_row, 0)
            copy = np.delete(copy, random_row, 0)

        test_data = np.array(test_data)
        train_data = np.array(train_data)
        test_class = np.array(test_class).astype(int).flatten()
        train_class = np.array(train_class).astype(int).flatten()

        correct = 0
        wrong = 0

        # do knn classifier on original dataset
        for i in range(len(test_data)):
            test = test_data[i]
            dist_list = []
            k_class = []
            for i in range(len(train_data)):
                train = train_data[i]
                dist_list.append(dist(test, train, train_class[i]))
            dist_list.sort(key = lambda x: x[0]) # sort by lowest distance 
            dist_list = dist_list[:k] # cut to k nearest neighbours

            # extract the class value
            for x in dist_list:
                k_class.append(x[1])

            # determine if majority is 0 or 1
            m = stats.mode(k_class)

             # determine if classification is correct or wrong
            if m[0][0] == test_class[i]:
                correct += 1
            else:
                wrong += 1

        accuracy = correct/(correct+wrong)
        pre_accuracy.append(accuracy)

        correct = 0
        wrong = 0
        # perform fisher linear discriminant on training data

        # split training data based on the class it belongs too
        zero = []
        one = []
        k_class = []
        for i in range(len(train_data)):
            if train_class[i] == 0:
                zero.append(train_data[i])
            else:
                one.append(train_data[i])

        zero = np.array(zero)
        one = np.array(one)

        # compute the mean for each class
        mean_zero = np.mean(zero, axis=0)
        mean_one = np.mean(one, axis=0)

        # compute within-class scatter matricies of each class
        scatter_zero = scatter_matrix(zero, mean_zero)
        scatter_one = scatter_matrix(one, mean_one)

        scatter = scatter_zero + scatter_one
        #inverse
        scat_inv = np.linalg.inv(scatter)

        # compute the optimal line direction
        v = np.matmul(scat_inv, np.array(mean_zero - mean_one))

        # project all the data onto one dimension for training and test data
        train_proj = []
        for i in range(len(train_data)):
            y = np.matmul(v, train_data[i])
            train_proj.append(y)

        test_proj = []
        for i in range(len(test_data)):
            y = np.matmul(v, test_data[i])
            test_proj.append(y)

        # perform knn on the reduced dimension data set
        for i in range(len(test_proj)):
            dist_list = []
            test = test_proj[i]
            dist_list = []
            k_class = []
            for i in range(len(train_proj)):
                train = train_proj[i]
                dist_list.append(dist_FLD(test, train, train_class[i]))
            dist_list.sort(key = lambda x: x[0]) # sort by lowest distance 
            dist_list = dist_list[:k] # cut to k nearest neighbours

            # extract the class value
            for x in dist_list:
                k_class.append(x[1])

            # determine if majority is 0 or 1
            m = stats.mode(k_class)

             # determine if classification is correct or wrong
            if m[0][0] == test_class[i]:
                correct += 1
            else:
                wrong += 1
            
        accuracy = correct/(correct+wrong)
        post_accuracy.append(accuracy)
        print("Trial " + str(loop+1) + " complete...")

    # Compute mean and standard deviation from the 10 trials for original dimensions
    pre_mean = np.mean(pre_accuracy)
    pre_std_dev = np.std(pre_accuracy)

    print("\nResults for K Nearest Neighbour Classifier for orignal dimensions for k=" + str(k))
    for i in range(len(pre_accuracy)):
        print("Accuracy for trial " + str(i+1) + ": " + str(round(100*pre_accuracy[i], 2)) + "%")
    print("\nMean: " + str(pre_mean))
    print("Standard Deviation: " + str(pre_std_dev))

    # compute mean and standard deviation from the 10 trials for reduced dimensions
    post_mean = np.mean(post_accuracy)
    post_std_dev = np.std(post_accuracy)

    print("\nResults for K Nearest Neighbour Classifier after applying FLD for k=" + str(k))
    for i in range(len(post_accuracy)):
        print("Accuracy for trial " + str(i+1) + ": " + str(round(100*post_accuracy[i], 2)) + "%")
    print("\nMean: " + str(post_mean))
    print("Standard Deviation: " + str(post_std_dev))

    # the difference between means to see if knn classifier accuracy changed
    diff = abs(post_mean - pre_mean)
    print("Difference between means: " + str(diff))

# whatever function calls you want to do
def calls():
    func(5)

calls()

# END: Ishaan Patel