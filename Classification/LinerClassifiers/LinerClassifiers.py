from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']


def loadData():
    # 读取数据
    localPath = os.path.dirname(__file__)
    filePath = os.path.join(localPath, "breast-cancer-wisconsin.data")
    data = pd.read_csv(filePath, names=column_names)

    # 替换、删除缺失值
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    print(data.shape)
    return data


def main():
    data = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(
        data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
    print(Y_train.value_counts())
    print(Y_test.value_counts())

    # 数据标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # 训练逻辑斯蒂模型，采用默认参数
    lr = LogisticRegression()
    sgdc = SGDClassifier()
    lr.fit(X_train, Y_train)
    lr_Y_predict = lr.predict(X_test)

    sgdc.fit(X_train, Y_train)
    sgdc_Y_predict = sgdc.predict(X_test)

    # 分析模型性能
    print('Accuracy of LR Classifier:', lr.score(X_test, Y_test))
    print(classification_report(Y_test, lr_Y_predict,
                                target_names=['Benign', 'Malignant']))

    print('Accuracy of SGDC Classifier:', sgdc.score(X_test, Y_test))
    print(classification_report(Y_test, sgdc_Y_predict,
                                target_names=['Benign', 'Malignant']))


if __name__ == "__main__":
    main()
