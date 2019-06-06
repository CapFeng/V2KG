import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def loadData():
    # 导入数据
    localPath = os.path.dirname(__file__)
    filePath = os.path.join(localPath, "titanic.txt")
    titanic = pd.read_csv(filePath)
    print(titanic.head())
    titanic.info()
    # 数据补全
    X = titanic[["pclass", "age", "sex"]]
    Y = titanic["survived"]
    X.info()
    X["age"].fillna(X["age"].mean(), inplace=True)
    X.info()
    return X, Y


def main():
    X, Y = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=33
    )
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train.to_dict(orient="record"))
    print(vec.feature_names_)
    X_test = vec.fit_transform(X_test.to_dict(orient="record"))
    # 训练模型
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    Y_predict = dtc.predict(X_test)
    # 评价
    print(dtc.score(X_test, Y_test))
    print(classification_report(Y_predict, Y_test,
                                target_names=["dead", "survived"]))


if __name__ == "__main__":
    main()
