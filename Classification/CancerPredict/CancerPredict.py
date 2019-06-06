from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def loadData():
    localPath = os.path.dirname(__file__)
    train = os.path.join(localPath, "breast-cancer-train.csv")
    test = os.path.join(localPath, "breast-cancer-test.csv")
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_test_negative = df_test.loc[df_test['Type']
                                   == 0][['Clump Thickness', 'Cell Size']]
    df_test_positive = df_test.loc[df_test['Type']
                                   == 1][['Clump Thickness', 'Cell Size']]
    return df_test_negative, df_test_positive, df_train, df_test


def main():
    df_test_negative, df_test_positive, df_train, df_test = loadData()
    plt.scatter(df_test_negative['Clump Thickness'],
                df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'],
                df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()

    intercept = np.random.random([1])
    coef = np.random.random([2])
    lx = np.arange(0, 12)
    ly = (-intercept-lx*coef[0])/coef[1]
    plt.plot(lx, ly, c='yellow')

    plt.scatter(df_test_negative['Clump Thickness'],
                df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'],
                df_test_positive['Cell Size'], marker='x', s=150, c='black')

    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')

    plt.show()

    # 导入逻辑斯蒂回归分类器
    lr = LogisticRegression()
    lr.fit(df_train[['Clump Thickness', 'Cell Size']]
           [:10], df_train['Type'][:10])
    print('Testing accuracy (10 training sample):', lr.score(
        df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

    # 绘制前十次训练后的图
    intercept = lr.intercept_
    coef = lr.coef_[0, :]
    ly = (-intercept-lx*coef[0])/coef[1]

    plt.plot(lx, ly, c='green')
    plt.scatter(df_test_negative['Clump Thickness'],
                df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'],
                df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()

    # # 使用全部数据进行训练
    lr = LogisticRegression()
    lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
    print('Testing accuracy (all training sample):%f' % lr.score(
        df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

    # 绘制全部数据训练后的图
    intercept = lr.intercept_
    coef = lr.coef_[0, :]
    ly = (-intercept-lx*coef[0])/coef[1]

    plt.plot(lx, ly, c='blue')
    plt.scatter(df_test_negative['Clump Thickness'],
                df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'],
                df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()


if __name__ == "__main__":
    main()
