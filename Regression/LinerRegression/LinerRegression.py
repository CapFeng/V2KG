
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


def loadData():
    boston = load_boston()
    print(boston.DESCR)
    return boston


def main():
    boston = loadData()
    X = boston.data
    Y = boston.target
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=33)
    print('The max target value is', np.max(boston.target))
    print('The min target value is ', np.min(boston.target))
    print('The average target value is ', np.mean(boston.target))

    # 对数据进行标准化处理
    ss_X = StandardScaler()
    ss_Y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    Y_train = ss_Y.fit_transform(Y_train.reshape(-1, 1))
    Y_test = ss_Y.transform(Y_test.reshape(-1, 1))

    # 导入线性回归模型并训练
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    lr_Y_predict = lr.predict(X_test)

    # 导入SDGR模型并训练
    sgdr = SGDRegressor()
    sgdr.fit(X_train, Y_train.ravel())
    sgdr_Y_predict = sgdr.predict(X_test)

    # 评估模型性能 进行对比 发现模型自带评价score等价于r2_score
    print('-----------------------------------------------------------------------')
    print('The value of default measurement of LinerRegression is ',
          lr.score(X_test, Y_test))
    print('The value of R-squared of LinerRegression is ',
          r2_score(Y_test, lr_Y_predict))
    print('The mean squared error of LinerRegression is ', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_Y_predict)))
    print('The mean absolute error of LinerRegression is ', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(lr_Y_predict)))
    print('-----------------------------------------------------------------------')
    print('The value of default measurement of SGDRession is ',
          sgdr.score(X_test, Y_test))
    print('The value of R-squared of SGDRession is ',
          r2_score(Y_test, sgdr_Y_predict))
    print('The mean squared error of SGDRession is ', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_Y_predict)))
    print('The mean absolute error of SGDRession is ', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(sgdr_Y_predict)))

    # SVM Regression
    # 线性核函数SVR
    liner_svr = SVR(kernel='linear')
    liner_svr.fit(X_train, Y_train.ravel())
    liner_svr_y_predict = liner_svr.predict(X_test)

    # 多项式核函数SVR
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(X_train, Y_train.ravel())
    poly_svr_y_predict = poly_svr.predict(X_test)

    # 径向基核函数SVR
    rbf_svr = SVR(kernel="rbf")
    rbf_svr.fit(X_train, Y_train.ravel())
    rbf_svr_y_predict = rbf_svr.predict(X_test)

    # 对三种核函数的SVR进行性能评估
    print('-----------------------------------------------------------------------')
    print('R-square value of linear SVR is:', liner_svr.score(X_test, Y_test))
    print('The MSE of linear SVR is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(liner_svr_y_predict)))
    print('The MAE of linear SVR is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(liner_svr_y_predict)))
    print('-----------------------------------------------------------------------')
    print('R-square value of poly SVR is:', poly_svr.score(X_test, Y_test))
    print('The MSE of poly SVR is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_svr_y_predict)))
    print('The MAE of poly SVR is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(poly_svr_y_predict)))
    print('-----------------------------------------------------------------------')
    print('R-square value of rbf SVR is:', rbf_svr.score(X_test, Y_test))
    print('The MSE of rbf SVR is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_svr_y_predict)))
    print('The MAE of rbf SVR is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rbf_svr_y_predict)))

    # 两种K近邻模型
    # 预测方式：平均回归
    uni_knr = KNeighborsRegressor(weights='uniform')
    uni_knr.fit(X_train, Y_train.ravel())
    uni_knr_y_predicrt = uni_knr.predict(X_test)

    # 预测方式：距离加权
    dis_knr = KNeighborsRegressor(weights='distance')
    dis_knr.fit(X_train, Y_train.ravel())
    dis_knr_y_predict = dis_knr.predict(X_test)

    # 对两种k近邻模型进行性能评估
    print('-----------------------------------------------------------------------')
    print('R-square value of uniform-weighted KNR is:',
          uni_knr.score(X_test, Y_test))
    print('The MSE of uniform-weighted KNR is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_y_predicrt)))
    print('The MAE of uniform-weighted KNR is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(uni_knr_y_predicrt)))
    print('-----------------------------------------------------------------------')
    print('R-square value of distance-weighted KNR is:',
          dis_knr.score(X_test, Y_test))
    print('The MSE of distance-weighted KNR is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_y_predict)))
    print('The MAE of distance-weighted KNR is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dis_knr_y_predict)))

    # 使用回归树模型
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, Y_train.ravel())
    dtr_y_predict = dtr.predict(X_test)

    # 对回归树进行性能评估
    print('-----------------------------------------------------------------------')
    print('R-square value of DecisionTreeRegressor is:', dtr.score(X_test, Y_test))
    print('The MSE of DecisionTreeRegressor is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_y_predict)))
    print('The MAE of DecisionTreeRegressor is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(dtr_y_predict)))

    # 使用三种集成模型进行训练
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train.ravel())
    rfr_y_predict = rfr.predict(X_test)

    etr = ExtraTreesRegressor()
    etr.fit(X_train, Y_train.ravel())
    etg_y_predict = etr.predict(X_test)

    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, Y_train.ravel())
    gbr_y_predict = gbr.predict(X_test)

    # 对三种集成模型进行性能评估
    print('-----------------------------------------------------------------------')
    print('R-square value of RandomForestRegressor is:', rfr.score(X_test, Y_test))
    print('The MSE of RandomForestRegressor is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_y_predict)))
    print('The MAE of RandomForestRegressor is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(rfr_y_predict)))

    print('-----------------------------------------------------------------------')
    print('R-square value of ExtraTreesRegressor is:', etr.score(X_test, Y_test))
    print('The MSE of ExtraTreesRegressor is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etg_y_predict)))
    print('The MAE of ExtraTreesRegressor is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(etg_y_predict)))
    print(np.sort(list(zip(etr.feature_importances_, boston.feature_names)), axis=0))
    print('-----------------------------------------------------------------------')
    print('R-square value of GradientBoostingRegressor is:',
          gbr.score(X_test, Y_test))
    print('The MSE of GradientBoostingRegressor is:', mean_squared_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbr_y_predict)))
    print('The MAE of GradientBoostingRegressor is:', mean_absolute_error(
        ss_Y.inverse_transform(Y_test), ss_Y.inverse_transform(gbr_y_predict)))


if __name__ == "__main__":
    main()
