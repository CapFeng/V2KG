from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


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

    # 转化特征为特征向量
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.fit_transform(X_test.to_dict(orient='record'))

    # 单一决策树
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, Y_train)
    dtc_y_predict = dtc.predict(X_test)

    # 随机森林分类器
    rfc = RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    rfc_y_predict = rfc.predict(X_test)

    # 梯度提升决策树
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, Y_train)
    gbc_y_predict = gbc.predict(X_test)

    # 集成模型的预测性能
    print('The accuracy of decision tree is ', dtc.score(X_test, Y_test))
    print(classification_report(dtc_y_predict, Y_test))

    print('The accuracy of random forest classifier is ',
          rfc.score(X_test, Y_test))
    print(classification_report(rfc_y_predict, Y_test))

    print('The accuracy of gradient tree boosting is ', gbc.score(X_test, Y_test))
    print(classification_report(gbc_y_predict, Y_test))


if __name__ == "__main__":
    main()
