from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import pandas as pd
from nltk import tokenize
from methods import *

sid = SentimentIntensityAnalyzer()


#On training dataset

test_pos=[]
tp=0
fn=0
with open("data/test-pos.txt", "r") as f:
        for i, line in enumerate(f):
            line = remove_non_ascii_2(line)
            line = cleanhtml(line)
            words =[term for term in preprocess(line) if term not in stop]
            line = (' ').join(words)
            polarity =sid.polarity_scores(line)
            if -0.2<polarity['compound']<=1:
                test_pos.append('pos')
                tp+=1
            else:
                test_pos.append('neg')
                fn+=1
            
# print test_pos
#print tp
#print fn

test_neg=[]
tn=0
fp=0

with open("data/test-neg.txt", "r") as f:
        for i, line in enumerate(f):
            line = remove_non_ascii_2(line)
            line = cleanhtml(line)
            words =[term for term in preprocess(line) if term not in stop]
            line = (' ').join(words)
            polarity =sid.polarity_scores(line)
            if 0<polarity['compound']<=1:
                test_neg.append('pos')
                fp+=1
            else:
                test_neg.append('neg')
                tn+=1
            
# print test_neg
#print fp
#print tn

accuracy=float((tp+tn))/(len(test_pos)+len(test_neg))

print "predicted:\tpos\tneg"
print "actual:"
print "pos\t\t%d\t%d" % (tp, fn)
print "neg\t\t%d\t%d" % (fp, tn)
        
print "accuracy: %f" % (accuracy)



print 'Analysis for Demonetization Twitter Data'

tweets=pd.read_csv("data/stream/demonetization-tweets.csv",encoding = "ISO-8859-1")

'''To read tweets from data/stream/json files

with open('data/stream/stream_demonitization10.json', 'rb') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)

data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
tweets = pd.read_json(data_json_str)
#tweets[:10]


'''

tweets['sentiment_compound_polarity']=tweets.text.apply(lambda x:sid.polarity_scores(x)['compound'])
tweets['sentiment_neutral']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neu'])
tweets['sentiment_negative']=tweets.text.apply(lambda x:sid.polarity_scores(x)['neg'])
tweets['sentiment_pos']=tweets.text.apply(lambda x:sid.polarity_scores(x)['pos'])
tweets['sentiment_type']=''
tweets.loc[tweets.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
tweets.loc[tweets.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
tweets.loc[tweets.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

#print tweets.head()

pos=0
neg=0
neu=0
for i in range(len(tweets['text'])):
#     print tweets['text'][i]
#     print tweets['sentiment_type'][i]
    if tweets['sentiment_type'][i] == 'POSITIVE':
        pos +=1
    elif tweets['sentiment_type'][i] == 'NEGATIVE':
        neg +=1
    elif tweets['sentiment_type'][i] == 'NEUTRAL':
        neu +=1
#print 'Total Tweets', len(tweets['text'])
print 'Positive', pos
print 'Negative', neg
print 'Neutral', neu