
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def loadData():
    iris = load_iris()
    print(iris.data.shape)
    print(iris.DESCR)
    return iris


def main():
    iris = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=33)

    # 数据特征标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    # 使用K近邻算法进行训练预测
    knc = KNeighborsClassifier()
    knc.fit(X_train, Y_train)
    Y_predict = knc.predict(X_test)

    # 对模型进行评价
    print('Accuracy of K-Neighbors:', knc.score(X_test, Y_test))
    print(classification_report(Y_test, Y_predict, target_names=iris.target_names))


if __name__ == "__main__":
    main()
