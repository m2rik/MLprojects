import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print (cancer.keys())
print (cancer['DESCR'])
#30 different independent features
#reduce the 30 to 2 dimensions
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print (df.head(5))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
print (scaled_data)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(scaled_data.shape)
print (x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('1st principle component')
plt.ylabel('2nd principle component')
plt.show()