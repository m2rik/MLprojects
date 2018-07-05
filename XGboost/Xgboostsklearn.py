import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
#parallel computations on a single amchine
import matplotlib.pyplot as pyplot

dataset = pd.read_csv("Churn_modelling.csv")
print (dataset.head())
#exited column is the dependent variable Y all others are features(10)

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values #categorical 0/1
#encode the y(categorical data)\
#use a one hot encoder or label encoder 1/0
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
LabelEncoder_X_1=LabelEncoder()
X[:,1]=LabelEncoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
OneHotEncoder=OneHotEncoder(categorical_features=[1])
OneHotEncoder.fit_transform(X).toarray()
X=X[:,1] #dummy variable

#split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#fit xgboost-- we can fine tune more paramters and also user kfold methods to improve the accuracy
import xgboost 
classifier=xgboost.XGBClassifier()
classifier.fit(X_train,y_train)

#predict
y_pred=classifier.predict(X_test)


#can use confusion matrix to rpedict how well the prediction has been done
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#accuracy of the model
accuracy=accuracy_score(y_test,y_pred)
print (accuracy)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

