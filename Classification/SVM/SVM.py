from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def loadData():
    # 导入数据并划分测试集
    digits = load_digits()
    print(digits.data.shape)
    return digits


def main():
    digits = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=33)
    print(Y_train.shape)
    print(Y_test.shape)

    # 数据标准化并训练模型
    ss = StandardScaler()
    lsvc = LinearSVC()
    lsvc.fit(X_train, Y_train)
    Y_predict = lsvc.predict(X_test)

    print('Accuracy of LinerSVC:', lsvc.score(X_test, Y_test))
    print(classification_report(Y_test, Y_predict,
                                target_names=digits.target_names.astype(str)))


if __name__ == "__main__":
    main()
