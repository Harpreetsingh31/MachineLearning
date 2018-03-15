import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

#Importing dataset
df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = df['class']

#Initializing linear regressor
clf  = LogisticRegression()
clf1 = neighbors.KNeighborsClassifier()
clf2 = svm.SVC()

#For classification report
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .20,random_state = 42)
clf1.fit(X_train,y_train)
y_pred = clf1.predict(X_test)

#For parameter tuning for KNN
#knn = np.arange(1,50)
#parameters = {'knn__n_neighbors':knn}
#clf1_cv = GridSearchCV(clf1,parameters,cv=5)
#clf1_cv.fit(X,y)
#print("Tuned Logistic Regression Parameters: {}".format(clf1_cv.best_params_))

#Compute cross_validation for LogReg
cv_score = cross_val_score(clf,X,y,cv=5)
#print('CV Score: {}'.format((cv_score)))
print('Average 5_fold CV Score for LogReg: {}'.format(np.mean(cv_score)))

#Compute cross_validation for KNN
cv_score1 = cross_val_score(clf1,X,y,cv=5,scoring='accuracy')
#print('CV Score: {}'.format((cv_score1)))
print('Average 5_fold CV Score for KNN: {}'.format(np.mean(cv_score1)))
# Generate the confusion matrix and classification report

print(classification_report(y_test, y_pred))

#Compute cross_validation for SVM
cv_score2 = cross_val_score(clf2,X,y,cv=5,scoring='accuracy')
#print('CV Score: {}'.format((cv_score2)))
print('Average 5_fold CV Score for SVM: {}'.format(np.mean(cv_score2)))

#Applying Neural network
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import scale
y2 = pd.get_dummies(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y2,test_size = .20,random_state = 42)

model = Sequential()

model.add(Dense(9, input_dim=9, activation='relu'))

model.add(Dense(5, activation='relu'))

model.add(Dense(3, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop',metrics=['accuracy'])

history = model.fit(X_train,y_train,nb_epoch= 10, verbose=1,batch_size=5)

[test_loss, test_acc] = model.evaluate(X_test, y_test, batch_size=5)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
