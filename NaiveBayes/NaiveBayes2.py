import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
df=pd.read_csv('smsspam',sep='\t',names=['Status','Message'])
df.head()
len(df)
len(df[df.Status--'Spam'])
df.loc[df["Stutus"]=='ham',"Status"]=1
df.loc[df["Stutus"]=='spam',"Status"]=0

df.head()
df_x=df["Message"]
df_y=df["Status"]

#extract text data to numbers
#use count vectorizer to 
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=0)

#TF-IDF vectorizer
cv=TfidfVecotorizer(min_df=1,stop_words='english')
x_traincv=cv.fit_transform(["hi how are you im Sam","hey whats your name","hey my name is jesicca"])
x_traincv.toarray()
cv.get_feature_names()
#columns are the words-extract features out of text data,
cv=TfidfVecotorizer()
x_traincv=cv1.fit_transform(x_train)
a=x_traincv1.toarray()
#multinomial naive bayes
cv1.inverse_transform(a[0])#checking length we see that very few of them are 1
x_train.iloc[0]

mnb=MultinomialNB()
y_train=y_train.astype('int')
mnb.fit(x_train,y_train)
x_test=cv.transform(x_test)
pred=mnb.predict(x_test)
actual=np.array(y_test)
for i in range(len(pred)):
	if pred[1]==actual[1]:
		count=count+1




