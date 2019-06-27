import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def loadData():
    # 导入数据
    localPath = os.path.dirname(__file__)
    filePath = os.path.join(localPath, "titanic.txt")
    titanic = pd.read_csv(filePath)
    X = titanic[['pclass', 'age', 'sex']]
    Y = titanic['survived']
    # 数据补全
    X['age'].fillna(X['age'].mean(), inplace=True)
    return X, Y


def main():
    X, Y = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=33)
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))

    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    print('rfc accuracy', rfc.score(X_test, Y_test))

    xgbc = XGBClassifier()
    xgbc.fit(X_train, Y_train)
    print('xgbc accuracy', xgbc.score(X_test, Y_test))


if __name__ == "__main__":
    main()
