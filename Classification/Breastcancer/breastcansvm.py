import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, cross_validation, svm

#importing dataset
df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

#Creating training and test sets
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=.20,random_state = 42)

#initializing linear regressor
clf = svm.SVC()

#training linear regressor
clf.fit(X_train, y_train)

#Prediction on the test data
y_predict = clf.predict(X_test)

#correlation
print('Auc: {}'.format(clf.score(X_test, y_test)))
rsme = np.sqrt(mean_squared_error(y_test,y_predict))
print('Rmse: {}'.format(rsme))

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)
