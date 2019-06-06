
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups


def loadData():
    # 导入数据(14MB 无法下载  下载后的文件无法处理)
    news = fetch_20newsgroups(subset='all')
    print(len(news.data))
    print(news.data[0])
    return news


def main():
    news = loadData()
    # 数据分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        news.data, news.target, test_size=0.25, random_state=33)

    # 特征抽取文本特征转化为特征向量
    vec = CountVectorizer()
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)

    # 训练模型
    mnb = MultinomialNB()
    mnb.fit(X_train, Y_train)
    y_preditc = mnb.predict(X_test)
    print('Accuracy of Naive Bayes:', mnb.score(X_test, Y_test))
    print(classification_report(Y_test, y_preditc, target_names=news.target_names))


if __name__ == "__main__":
    main()
