import pandas as pd
import numpy as np
import os
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
from sklearn import feature_selection


def loadData():
    # 导入数据
    localPath = os.path.dirname(__file__)
    filePath = os.path.join(localPath, "titanic.txt")
    titanic = pd.read_csv(filePath)
    # 数据补全
    X = titanic.drop(['row.names', 'name', 'survived'], axis=1)
    Y = titanic["survived"]
    X["age"].fillna(X["age"].mean(), inplace=True)
    X.fillna('UNKONWN', inplace=True)
    return X, Y


def main():
    X, Y = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=33)
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))
    print(len(vec.feature_names_))
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train, Y_train)
    print(dt.score(X_test, Y_test))

    fs = feature_selection.SelectPercentile(
        feature_selection.chi2, percentile=20)
    X_train_fs = fs.fit_transform(X_train, Y_train)
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train_fs, Y_train)
    X_test_fs = fs.transform(X_test)
    print(dt.score(X_test_fs, Y_test))

    percentiles = range(1, 100, 2)
    results = []
    for i in percentiles:
        fs = feature_selection.SelectPercentile(
            feature_selection.chi2, percentile=i)
        X_test_fs = fs.fit_transform(X_train, Y_train)
        scores = cross_val_score(dt, X_train_fs, Y_train, cv=5)
        results = np.append(results, scores.mean())
    print(results)

    opt = np.where(results == results.max())[0]
    print(opt)
    print('Optimal number of features %d' % percentiles[opt[0]])
    pl.plot(percentiles, results)
    pl.xlabel('percent of features')
    pl.ylabel('accuracy')
    pl.show()

    fs = feature_selection.SelectPercentile(
        feature_selection.chi2, percentile=percentiles[opt[0]])
    X_train_fs = fs.fit_transform(X_train, Y_train)
    # dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train_fs, Y_train)
    X_test_fs = fs.transform(X_test)
    print(dt.score(X_test_fs, Y_test))


if __name__ == "__main__":
    main()
