import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
#dataset mall-customers
#problem - find spending score btw 1-100
D=pd.read_csv("Customers.csv")
X=D.iloc[:,[3,4]].values
#how many clusters are needed? use the elbow methodto find optimal no. of clusters
from sklearn.cluster import KMeans
wcss=[]#within cluster sum of squares
for i in range(1,11):
	kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
	kmeans.fit(X)#since independent features
	wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow")
plt.xlabel("no. of clusters")
plt.ylabel("wcss")
plt.show()
#fit the kmeans using k=5 as found out

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X) #all clusters in y_means

#visualize the clusters
#whichever belongs to 0 group them as 1 and similarly scatter for y also
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='cluster1') #error--IndexError: index 1 is out of bounds for axis 1 with size 1
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='cluster5')

#display centroids of each clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Mall spending customers')
plt.xlabel('annualIncome($)')
plt.ylabel('Spendingscore(1-100)')
plt.legend()
plt.show() #centroids show the mean 


