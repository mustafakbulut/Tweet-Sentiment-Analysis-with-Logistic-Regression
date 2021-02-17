from flask import Flask,render_template,request
import pickle
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
nltk.download('twitter_samples')
nltk.download('stopwords')
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $  
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove retweets
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word 
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True) #preserve_case=False means downcasing everything except for emoticons
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_processed = []
    for word in tweet_tokens:
        if ((word not in stopwords_english or word=='not')  and  word not in string.punctuation):  
            stemmed_word = stemmer.stem(word)  # stemming word
            tweets_processed.append(stemmed_word)
    return tweets_processed
def extract_features(tweet, freqs):
    tweet_processed = process_tweet(tweet)
    x = np.zeros((1, 3)) 
    x[0,0] = 1 #bias
    for word in tweet_processed:
        x[0,1] += freqs.get((word, 1.0),0)
        x[0,2] += freqs.get((word, 0.0),0)
    assert(x.shape == (1, 3))
    return x
def predict_tweet(tweet, freqs, model):
    x = extract_features(tweet,freqs)
    y_pred = model.predict(x)
    return y_pred
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]
test_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[4000:]
tweets = load_dataset("tweets.csv", ['text']) # dataset from kaggle
for i in range (0,10000):
    train_neg.append(tweets['text'][i])
for i in range (10000,20000):
    train_pos.append(tweets['text'][i])
train_x = train_pos + train_neg 
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_x = test_pos + test_neg
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
freqs = build_freqs(train_x, train_y)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)
Y = train_y
Y.shape = (Y.shape[0],)

app= Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    tweet=request.form.get("tweet")
    filename='LogisticRegression_model.sav'
    model=pickle.load(open(filename, 'rb'))
    predict=predict_tweet(tweet,freqs,model)
    if(predict==1.0):
        text="Positive Sentiment"
    else:
        text="Negative Sentiment"
    return render_template("index.html",text=text)
@app.route("/train",methods=['POST'])
def train():
    LogisticRegression(random_state=0).fit(X, Y) #training
    return render_template("index.html",text="Succesfully trained.")
@app.route("/test",methods=['POST'])
def test(): #testing
    X_test = np.zeros((len(test_x), 3))
    for i in range(len(test_x)):
        X_test[i, :]= extract_features(test_x[i], freqs)
    Y_test = test_y
    Y_test.shape = (Y_test.shape[0],)
    filename='LogisticRegression_model.sav'
    model=pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X_test)
    text="Accuracy= %"+str(accuracy_score(Y_test,y_pred)*100)
    return render_template("index.html",text=text)
if __name__=="__main__":
    app.run()