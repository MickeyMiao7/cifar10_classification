#!/usr/bin/python
import numpy as np
from svd import unpickle

dimension = 200
k = 5
normalize = False
gray = True

labels_train = []
for i in xrange(1, 6):
    temp = unpickle("../cifar-10-batches-py/data_batch_%s" % str(i))["labels"] 
    labels_train += temp

data_train = np.load("../cifar-10-batches-py/gray_data_train.npy") if gray else np.load("../cifar-10-batches-py/data_train.npy")
u_train = np.load("../cifar-10-batches-py/gray_u_train.npy") if gray else np.load("../cifar-10-batches-py/u_train.npy")
s_train = np.load("../cifar-10-batches-py/gray_s_train.npy") if gray else np.load("../cifar-10-batches-py/s_train.npy")
v_train = np.load("../cifar-10-batches-py/gray_v_train.npy") if gray else np.load("../cifar-10-batches-py/v_train.npy")


#transformed_data_train = np.dot(np.transpose(u_train[:, :dimension]), data_train)
transformed_data_train = np.dot(u_train[:, :dimension].T, data_train)
if normalize:
    transformed_data_train = transformed_data_train / np.linalg.norm(transformed_data_train)



labels_test = unpickle("../cifar-10-batches-py/test_batch" )["labels"] 
data_test = np.load("../cifar-10-batches-py/gray_data_test.npy") if gray else np.load("../cifar-10-batches-py/data_test.npy")
transformed_data_test = np.dot(u_train[:, :dimension].T, data_test)


def knn(input, dataset, labels, k):  
    total = dataset.shape[0] # shape[0] stands for the num of row  
    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    diff = np.tile(input, (total, 1)) - dataset # Subtract element-wise  

    
    squared_diff = diff ** 2 # squared for the subtract  
    squared_dist = np.sum(squared_diff, axis = 1) # sum is performed by row  
    distance = squared_dist ** 0.5  
     
    # distance = np.sum(np.absolute(diff), axis=1)
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sorted_distIndices = np.argsort(distance)  
  
    class_count = {} # define a dictionary (can be append element)  
    for i in xrange(k):  
        ## step 3: choose the min k distance  
        vote_label = labels[sorted_distIndices[i]]  
  
        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        class_count[vote_label] = class_count.get(vote_label, 0) + 1  
    ## step 5: the max voted class will return  

    maxCount = 0  
    for key, value in class_count.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
    return maxIndex   


correct = 0
all = len(labels_test)
index = 0

"""
# Use linear classifier to classify
from sklearn import linear_model
clf = linear_model.SGDClassifier()
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
clf.fit(data_train.T, labels_train)
    
labels_predict = clf.predict(data_test.T)

for i in xrange(0, labels_predict.shape[0]):
    if labels_predict[i] == labels_test[i]:
        correct += 1
    index += 1
    print index, float(correct) / index
print "Use random projection and linear classifier"
print "When D = %d, gray = %s, Normalization(After): %s, Predict correct rate: %f" % (dimension, str(gray), str(normalize), float(correct) / index)



"""
# Use SVD and KNN
for i in xrange(all):
    testcase = transformed_data_test.T[i]
    predict_label = knn(testcase, transformed_data_train.T, labels_train, k)
    real_label= labels_test[i]
    index += 1
    if predict_label == real_label:
        correct += 1
    print index, float(correct)/ index
print "Use SVD and KNN"
print "When D=%d, k = %d, Normalization(After): %s, Total test case: %d, Predict correct rate: %d" %(dimension, k, str(normalize), all, float(correct) / index) 



