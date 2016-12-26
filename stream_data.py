import tweepy
from textblob import TextBlob
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import string
import operator 
import json
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk import bigrams
from collections import defaultdict
import vincent
import pandas


ckey= 'a7nsg5VRVno2sYVqFdAqlzNbf'
csecret= 'IvtGVKRdSqcfZl3bjYxUlxXtp2BjpHtJ74ldnF9IBr5qSm7nwb'

atoken= '92489553-BMtum2DRXwaqeYQ4bQlRx7fPZDdNkzzIOHTDEhpto'
asecret= 'xdtGtBlg1JiUdSWeiWSvdd6QCSeku2OoJAvrJs8lxz7WQ'

auth = tweepy.OAuthHandler(ckey,csecret)
auth.set_access_token(atoken, asecret)


api = tweepy.API(auth)

public_tweets = api.search('demonitization')

class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self,data_dir,query):
        self.outfile = "%s/stream_%s.json" % (data_dir, query)

    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:
                f.write(data)
                print(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        return True
    
data_dir = 'data/stream'
query="demonitization11"
twitter_stream =Stream(auth, MyListener(data_dir,query))
twitter_stream.filter(track=['#IndiaFightsCorruption', '#ModiDemonetisationCircus','IndiaDefeatsBlackMoney', 'Demometisation'],languages=['en'])