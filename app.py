# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:19:50 2020

@author: Trupti
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
model1 = pickle.load(open(filename, 'rb'))
cvector=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
#	df= pd.read_csv("covid.csv", encoding="latin-1")
#	df.drop(['Unnamed: 0'], axis=1, inplace=True)
#	# Features and Labels
#   sentiment = SentimentIntensityAnalyzer()
#   tweet_df['sentiment'] = tweet_df['tweet_lemmatized'].apply(lambda x: sentiment.polarity_scores(x))
#   tweet_df.head()
#
#   def sentiment_analyzer_scores(text):
#       score = sentiment.polarity_scores(text)
#       lb = score['compound']
#       if lb >= 0.05:
#           return 'Positive'
#       elif (lb > -0.05) and (lb < 0.05):
#           return 'Neutral'
#       else:
#           return 'Negative'

#    tweet_df['vader_sentiment'] = tweet_df['tweet_lemmatized'].apply(sentiment_analyzer_scores)
#	
#	# Extract Feature With CountVectorizer
#	cvector = CountVectorizer(min_df = 0.0, max_df = 1.0, ngram_range=(1,2))
#	bow = cvector.fit_transform(bow)  # Fit the Data
#    
#   pickle.dump(cvector, open('transform.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.20, random_state = 1)
#	#Logistic regression classifier 
#   from sklearn.feature_extraction.text import TfidfVectorizer
#	from sklearn.linear_model import LogisticRegression
#
#	tvec = CountVectorizer(min_df = 2, ngram_range = (1,3))
#   clf2 = LogisticRegression(solver = "lbfgs")

#Pipeline is used to compile the transformer and vectorizer together for the ‘Text_Feature’
#   from sklearn.pipeline import Pipeline
#   model1 = Pipeline([('vectorizer',tvec),('classifier',clf2)])

#   model1.fit(IV_train, DV_train)
#   predictions = model1.predict(IV_test)
#   filename = 'nlp_model.pkl'
#   pickle.dump(model1, open(filename, 'wb'))
    
	#Alternative Usage of Saved Model
	# joblib.dump(model1, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cvector.transform(data).toarray()
		my_prediction = model1.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True, use_reloader=False)
