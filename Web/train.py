
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

df=pd.read_csv('samsu.csv')
df.Open=df.Open.fillna(df.Open.mean())
df.Close=df.Close.fillna(df.Close.mean())

Dat=list(pd.DatetimeIndex(df['Date']).day)
Dat=np.reshape(Dat,(len(Dat),1))
OpenPrice=list(df['Open'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Day']=df['Date'].dt.dayofweek
df['Month']=pd.DatetimeIndex(df['Date']).month
df['Year']=pd.DatetimeIndex(df['Date']).year
df['Date']=Dat
df['Month']=pd.DatetimeIndex(df['Date']).month

df.drop(['High','Low','Adj Close','Volume'],axis=1,inplace=True);

Y=df['Close'].values
X=df.drop('Close',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

# from sklearn.linear_model import LinearRegression
#
# clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, y_train)
# pickle.dump(clfreg,open('model.pkl','wb'))
# y_pred=clfreg.predict(X_test)
# confidencereg = clfreg.score(X_test, y_test)
# print(confidencereg)

from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
clf.fit(X_train,y_train)
pickle.dump(clf,open('model.pkl','wb'))
y_pred=clf.predict(X_test)
clf.score(X_test,y_test)
