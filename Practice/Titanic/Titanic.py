import pandas as pd
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


localPath = os.path.dirname(__file__)
train = pd.read_csv(os.path.join(localPath, 'Data/train.csv'))
test = pd.read_csv(os.path.join(localPath, 'Data/test.csv'))
print(train.info())
print(test.info())

selected_feature = ['Pclass', 'Sex', 'Age',
                    'Embarked', 'SibSp', 'Parch', 'Fare']
x_train = train[selected_feature]
x_test = test[selected_feature]
y_train = train['Survived']

print(x_train['Embarked'].value_counts())
print(x_test['Embarked'].value_counts())
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
print(x_train.info())
print(x_test.info())

dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
x_test = dict_vec.transform(x_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
xgbc = XGBClassifier()
rfc_score = cross_val_score(rfc, x_train, y_train, cv=5)
xgbc_score = cross_val_score(xgbc, x_train, y_train, cv=5)
print(rfc_score.mean())
print(xgbc_score.mean())

rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
rfc_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv(os.path.join(
    localPath, 'Submission/rfc_submission.csv'), index=False)

xgbc.fit(x_train, y_train)
xgbc_y_predict = xgbc.predict(x_test)
xgbc_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv(os.path.join(
    localPath, 'Submission/xgbc_submission.csv'), index=False)


params = {'max_depth': range(2, 7), 'n_estimators': range(
    100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}

xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

xgbc_best_y_predict = gs.predict(x_test)
xgbc_best_submission = pd.DataFrame(
    {'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv(os.path.join(
    localPath, 'Submission/xgbc_best_submission.csv'), index=False)
