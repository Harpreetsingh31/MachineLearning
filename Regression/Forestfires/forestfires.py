import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import PolynomialFeatures,scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Importing dataset
df = pd.read_csv('forestfires.csv')
df.drop(['X','Y','month','day','FFMC','DMC','ISI','DC'],1,inplace=True)

#Inputs and Output
X = scale(np.array(df.drop(['area'],1)))
y = np.array(df.area.apply(lambda x: np.log(x+1)))

#Initializing linear regressor and KNN
clf  = Ridge()
##
##plt.scatter(X[:,1],y)
##plt.show()

##
##X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .20,random_state = 42)
###Compute cross_validation
##clf.fit(X_train,y_train)
##y_pred =clf.predict(X_test)
##cv_score = clf.score(X_test,y_test)
##coeff = clf.coef_
##inter = clf.intercept_
##print('CV Score: {}'.format(cv_score))
##rmse = np.sqrt(mean_squared_error(y_test,y_pred))
##print("Root Mean Squared Error: {}".format(rmse))
##
##print('Coeff: {}'.format(coeff))
##print('Intercept: {}'.format(inter))


##poly = PolynomialFeatures(degree = 1)
##X_new = poly.fit_transform(X)
##
##X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size = .20,random_state = 42)
###Compute cross_validation
##clf.fit(X_train,y_train)
##y_pred =clf.predict(X_test)
##cv_score = clf.score(X_test,y_test)
##print('CV Score with degree 6: {}'.format(cv_score))
##rmse = np.sqrt(mean_squared_error(y_test,y_pred))
##print("Root Mean Squared Error: {}".format(rmse))

###Compute cross_validation
##cv_score = cross_val_score(clf,X,y,cv=5)
##print('CV Score: {}'.format((cv_score)))
##print('Average 5_fold CV Score: {}'.format(np.mean(cv_score)))

#Applying Neural network
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
#y2 = pd.get_dummies(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .20,random_state = 42)

model = Sequential()

model.add(Dense(4, input_dim=4, activation='relu'))

model.add(Dense(5, activation='relu'))

model.add(Dense(3, activation='relu'))

model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer = 'adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,nb_epoch= 2000, verbose=1,batch_size=10)

[test_loss, test_acc] = model.evaluate(X_test, y_test, batch_size=10)
print("For epoch = 1000, and batch size of 10, the accuracy was 0.49 with loss of 3.49") 
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))



