
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

import re

import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

consumer_key='XXXXXXXXXXXXXXXXXXXXXX'
consumer_secret='XXXXXXXXXXXXXXXXXXXXXX'
access_token='XXXXXXXXXXXXXXXXXXXXXX'
access_token_secret='XXXXXXXXXXXXXXXXXXXXXX'
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True)

for ind in df.index:
    public_tweets = api.search('Microsoft', count=10, date=df['Date'][ind], lang='en')

    pol = []
    var = 0
    type(public_tweets)
    for tww in public_tweets:
        print(tww.text)
        analysis = TextBlob(tww.text)
        var = var + analysis.sentiment.polarity
    var = var / len(public_tweets)
    pol.append(var)


df['pol']=pol

Y=df['Close'].values
X=df.drop('Close',axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
clf.fit(X_train,y_train)
pickle.dump(clf,open('model.pkl','wb'))
y_pred=clf.predict(X_test)
clf.score(X_test,y_test)
