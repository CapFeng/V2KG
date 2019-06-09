import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def loadData():
    localPath = os.path.dirname(__file__)
    trainPath = os.path.join(localPath, 'optdigits.tra')
    testPath = os.path.join(localPath,'optdigits.tes')
    digits_train = pd.read_csv(trainPath, header=None)
    digits_test = pd.read_csv(testPath, header=None)
    return digits_train, digits_test

def plot_pca_scatter(x_digits, y_digits):
    estimator = PCA(n_components=2)
    x_pca = estimator.fit_transform(x_digits)
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = x_pca[:, 0][y_digits.as_matrix() == i]
        py = x_pca[:, 0][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.show()

def main():
    digits_train, digits_test = loadData()
    x_digits = digits_train[np.arange(64)]
    y_digits = digits_train[64]
    plot_pca_scatter(x_digits, y_digits)

    x_train = digits_train[np.arange(64)]
    y_train = digits_train[64]
    x_test = digits_test[np.arange(64)]
    y_test = digits_test[64]

    svc = LinearSVC()
    svc.fit(x_train, y_train)
    y_pre = svc.predict(x_test)

    estimator = PCA(n_components=20)
    pca_x_train = estimator.fit_transform(x_train)
    pac_x_test = estimator.transform(x_test)

    pca_svc = LinearSVC()
    pca_svc.fit(pca_x_train, y_train)
    pca_y_pre = pca_svc.predict(pac_x_test)

    print(svc.score(x_test, y_test))
    print(classification_report(y_test, y_pre, target_names=np.arange(10).astype(str),digits=4))
    
    print(pca_svc.score(pac_x_test, y_test))
    print(classification_report(y_test,pca_y_pre,target_names=np.arange(10).astype(str),digits=4))


if __name__ == "__main__":
    main()
