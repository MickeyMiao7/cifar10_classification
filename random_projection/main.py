#!/usr/bin/python

import numpy as np
import cPickle
from sklearn import random_projection

dimension = 200
gray = False
normalize = False
k = 5


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def get_data():
    if gray:
        data_train = np.load("../cifar-10-batches-py/gray_data_train.npy").T
        data_test = np.load("../cifar-10-batches-py/gray_data_test.npy").T
    else:
        data_train = unpickle("../cifar-10-batches-py/data_batch_1")["data"]
        for i in xrange(2, 6):
            temp = unpickle("../cifar-10-batches-py/data_batch_%s" % str(i))["data"] 
            data_train = np.vstack((data_train  ,temp))
        data_test = unpickle("../cifar-10-batches-py/test_batch" )["data"]
    # data_train = data_train.T

    labels_train = []
    for i in xrange(1, 6):
        temp = unpickle("../cifar-10-batches-py/data_batch_%s" % str(i))["labels"] 
        labels_train += temp
     
    # data_test = data_test.T
    labels_test = unpickle("../cifar-10-batches-py/test_batch" )["labels"]
    return data_train, labels_train, data_test, labels_test


def gaussian_random_projection(d):
    return random_projection.GaussianRandomProjection(n_components=d)


def sparse_random_projection(d):
    return random_projection.SparseRandomProjection(n_components=d)


def knn(input, dataset, labels, k):  
    total = dataset.shape[0] # shape[0] stands for the num of row  
    diff = np.tile(input, (total, 1)) - dataset # Subtract element-wise  
    
    squared_diff = diff ** 2 # squared for the subtract  
    squared_dist = np.sum(squared_diff, axis = 1) # sum is performed by row  
    distance = squared_dist ** 0.5  
    
    # distance = np.sum(np.absolute(diff), axis=1)
    sorted_distIndices = np.argsort(distance)  
  
    class_count = {} # define a dictionary (can be append element)  
    for i in xrange(k):  
        vote_label = labels[sorted_distIndices[i]]  
        class_count[vote_label] = class_count.get(vote_label, 0) + 1  
  
    maxCount = 0  
    for key, value in class_count.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
    # print class_count, maxIndex
    return maxIndex   


def main():
    data_train, labels_train, data_test, labels_test = get_data()

    transformer =  sparse_random_projection(dimension)
    data_train = transformer.fit_transform(data_train)
    data_test = transformer.transform(data_test)

    correct = 0
    index = 0
    
    # use linear classifier to classify
    from sklearn import linear_model
    clf = linear_model.SGDClassifier()
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    clf.fit(data_train, labels_train)
    
    labels_predict = clf.predict(data_test)

    for i in xrange(0, labels_predict.shape[0]):
        if labels_predict[i] == labels_test[i]:
            correct += 1
        index += 1
        print index, float(correct) / index
    print "Use random projection and linear classifier"
    print "When D = %d, gray = %s, Normalization(After): %s, Predict correct rate: %f" % (dimension, str(gray), str(normalize), float(correct) / index)

    
    """
    # use knn to classify
    test_cnt = len(labels_test)
    for i in xrange(test_cnt):
        testcase = data_test[i]
        predict_label = knn(testcase, data_train, labels_train, k)
        real_label= labels_test[i]
        index += 1
        if predict_label == real_label:
            correct += 1
        print index, float(correct) / index

    print "Use random projection and KNN"
    print "When D = %d, k = %d, gray = %s, Normalization(After): %s, Predict correct rate: %f" % (dimension, k, str(gray), str(normalize), float(correct) / index)

    """


if __name__ == "__main__":
    main()
