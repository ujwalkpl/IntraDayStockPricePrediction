import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import datetime
from datetime import date
import calendar
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    datea = request.form.values()

    output1 = next(datea)
    # public_tweets = api.search('Microsoft', count=10, date=, lang='en')
    #
    # pol = []
    # var = 0
    # type(public_tweets)
    # for tww in public_tweets:
    #     print(tww.text)
    #     analysis = TextBlob(tww.text)
    #     var = var + analysis.sentiment.polarity
    # var = var / len(public_tweets)




    datea = request.form.values()

    output1 = next(datea)
    a = list(map(int,output1.split('-')))

    def findDay(x,y,z):

        born = datetime.date(x, y, z)
        return born.strftime("%A")

        # Driver program


    day=findDay(a[0],a[1],a[2])
    output=day
    dict={'Sunday': 6,'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5}
    fday=dict[day]




    output2=int(next(datea))

    # final_features = [np.array(int_features)]
    prediction = model.predict([[a[2],output2,fday,a[1],a[0]]])

    output = round(prediction[0], 2)
    profit=round(output-output2)/(output2/100)

    # output = datea[0]
    return render_template('index.html', prediction_text='Closing price {}'.format(output),predictionprofit='Profit percentage :{}'.format(profit) if profit>0 else 'Percentage of Loss {}'.format((-1*profit)))
    # return render_template('index.html',prediction_text="Awesome")

if __name__ == "__main__":
    app.run(debug=True)
