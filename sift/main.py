import cv2
import numpy as np
import cPickle

train_all = False


def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data_train = np.load("../cifar-10-batches-py/gray_data_train.npy").T
data_train = data_train if train_all else data_train[: 10000, ]

sift = cv2.SIFT()
dense=cv2.FeatureDetector_create("Dense")
dense.setInt("initXyStep",1)
dense.setInt("initFeatureScale", 3)
img = data_train[0]
img = np.uint8(img)
kp = dense.detect(img)
kp, des=sift.compute(img,kp)
print len(kp)
