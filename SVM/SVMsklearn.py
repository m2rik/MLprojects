#classification algorithm
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#dataset problem- classify whether the person will purchase a product or not
#age/salary independent,purchase is the dependent variable
D=pd.read_csv("Social_Network_Ads.csv")
X=D.iloc[:,[2,3]].values
y=D.iloc[:,4].values#depedent variable
#maybe curved or linear line for classification
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScalar
sc=StandardScalar() #feature scaling[-2,+2]
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
y_pred=Classfier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#visualizing the SVM KERNELS
from matplotlib.colors import ListedColormap
X_set,y_set = X_test,y_test
X1,X2=np.


#also we can create a sample dataset by...
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=1000,n_features=20,n_informative=8,n_redundant=3,n_repeated=2,random_state=seed)