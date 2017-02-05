#!/usr/bin/python

import numpy as np
import cPickle


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def generate_gray_data():
    data_train = unpickle("../cifar-10-batches-py/data_batch_1")["data"]
    for i in xrange(2, 6):
        temp = unpickle("../cifar-10-batches-py/data_batch_%s" % str(i))["data"] 
        data_train = np.vstack((data_train  ,temp))
    gray_data_train = np.zeros((data_train.shape[0], data_train.shape[1] / 3), dtype=int)

    for i in xrange(0, data_train.shape[0]):
        for j in xrange(data_train.shape[1] / 3):
            gray_data_train[i, j] = (data_train[i,j] * 2989 + data_train[i, j + data_train.shape[1] / 3] * 5870 + data_train[i, j + 2 * data_train.shape[1] / 3] * 1140 + 5000) / 10000
            # print data_train[i,j], data_train[i, j + data_train.shape[1] / 3], data_train[i, j + 2 * data_train.shape[1] / 3], gray_data_train[i, j]
    np.save("../cifar-10-batches-py/gray_data_train", gray_data_train.T)
    # 50000 * 1024
    print gray_data_train.shape
    
    data_test = unpickle("../cifar-10-batches-py/test_batch")["data"]
    gray_data_test = np.zeros((data_test.shape[0], data_test.shape[1] / 3), dtype=int)

    for i in xrange(0, data_test.shape[0]):
        for j in xrange(data_test.shape[1] / 3):
            gray_data_test[i, j] = (data_test[i,j] * 2989 + data_test[i, j + data_test.shape[1] / 3] * 5870 + data_test[i, j + 2 * data_test.shape[1] / 3] * 1140 + 5000) / 10000
    np.save("../cifar-10-batches-py/gray_data_test", gray_data_test.T)
    # 10000 * 1024
    print gray_data_test.shape


def usv_train():
    data = unpickle("../cifar-10-batches-py/data_batch_1")["data"]
    for i in xrange(2, 6):
        temp = unpickle("../cifar-10-batches-py/data_batch_%s" % str(i))["data"] 
        data = np.vstack((data,temp))
    # 3072 * 50000
    data = np.transpose(data)
    np.save("data_train", data)

    u, s, v = np.linalg.svd(data, full_matrices=False)
    print u.shape, s.shape, v.shape
    np.save("../cifar-10-batches-py/u_train", u)
    np.save("../cifar-10-batches-py/s_train", s)
    np.save("../cifar-10-batches-py/v_train", v)


def gray_usv_train():
    gray_data = np.load("../cifar-10-batches-py/gray_data_train.npy")
    u, s, v = np.linalg.svd(gray_data, full_matrices=False)
    print u.shape, v.shape, s.shape
    np.save("../cifar-10-batches-py/gray_u_train", u)
    np.save("../cifar-10-batches-py/gray_s_train", s)
    np.save("../cifar-10-batches-py/gray_v_train", v)



def main():
    # generate_gray_data()
    # usv_train()
    gray_usv_train()


if __name__ == "__main__":
    main()
