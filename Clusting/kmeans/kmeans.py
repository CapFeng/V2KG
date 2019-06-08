# coding=utf-8
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import os

def loadData():
    localPath = os.path.dirname(__file__)
    trainPath = os.path.join(localPath, 'optdigits.tra')
    testPath = os.path.join(localPath,'optdigits.tes')
    digits_train = pd.read_csv(trainPath, header=None)
    digits_test = pd.read_csv(testPath, header=None)
    return digits_train, digits_test

def main():
    digits_train, digits_test = loadData()
    X_train = digits_train[np.arange(64)]
    Y_train = digits_train[64]
    X_test = digits_test[np.arange(64)]
    Y_test = digits_test[64]

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X_train)
    Y_pre = kmeans.predict(X_test)
    #使用ARI(Adjust Rand Index)进行KMeans的聚类性能评估
    print(metrics.adjusted_rand_score(Y_test, Y_pre))


if __name__ == "__main__":
    main()