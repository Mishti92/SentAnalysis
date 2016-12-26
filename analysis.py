## Analyzing the demonetization twitter data solution obtained using VADER method

import pandas as pd
from collections import Counter
from nltk import bigrams
from methods import *
from ggplot import *
#%matplotlib inline
import matplotlib.pyplot as plt
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


tweets=pd.read_csv("output/tweets_vader.csv",encoding = "ISO-8859-1")

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def wordcloud_by_province(tweets):
    stopwords = set(STOPWORDS)
    stopwords.add("https")
    stopwords.add("00A0")
    stopwords.add("00BD")
    stopwords.add("00B8")
    stopwords.add("ed")
    stopwords.add("demonetization")
    stopwords.add("Demonetization co")
    #Narendra Modi is the Prime minister of India
    stopwords.add("lakh")
    wordcloud = WordCloud(background_color="white",max_font_size=40, stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets['text_new'].str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Demonetization")
    
    
def polarity_distribution():
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
    #     else:
    #         print i
    print 'Total Tweets Analyzed', len(tweets['text'])
    print 'Positive', pos
    print 'Negative', neg
    print 'Neutral', neu
    print '******************************'



def process_data(tweets):
    tweets['text_new'] = ''
    tweets['tweetos'] = ''

    for i in range(len(tweets['text'])):
#        print i,tweets['text'][i]
        m = re.search('(?<=:)(.*)', tweets['text'][i])
        try:
            tweets['text_new'][i]=m.group(0)
        except AttributeError:
            tweets['text_new'][i]=tweets['text'][i]

    #print(tweets['text'].str.split(':')[0][0])  

    for i in range(len(tweets['text'])):
        try:
            tweets['tweetos'][i] = tweets['text'].str.split(':')[i][0]
        except AttributeError:    
            tweets['tweetos'][i] = 'other'
    return tweets

    
def term_occurence():
    #Counting  Most frequent hash tags, most common terms, bigrams used
    count_all = Counter()
    count_stop =Counter()
    count_single=Counter()
    count_hash=Counter()
    count_terms=Counter()
    count_bigrams = Counter()


    for i in range(len(tweets['text'])):
        text = tweets['text'][i]
        text = remove_non_ascii_2(text)
    #     print i, text
        terms_all = [term for term in preprocess(text)]
                    #filtered
        terms_stop = [term for term in preprocess(text) if term not in stop]
                    # Count terms only once, equivalent to Document Frequency
        terms_single = set(terms_all)
                    # Count hashtags only
        terms_hash = [term for term in preprocess(text) 
                          if term.startswith('#')]
                    # Count terms only (no hashtags, no mentions)
        terms_only = [term for term in preprocess(text) 
                          if term not in stop and
                          not term.startswith(('#', '@'))] 
                          # mind the ((double brackets))
                          # startswith() takes a tuple (not a list) if 
            #                we pass a list of inputs
                    #bigrams
        terms_bigram = bigrams(terms_stop)
    #                 # Update the counter
        count_all.update(terms_all)
        count_stop.update(terms_stop)
        count_single.update(terms_single)
        count_hash.update(terms_hash)
        count_terms.update(terms_only)
        count_bigrams.update(terms_bigram)
#    print 'All: ', count_all.most_common(10)
    print 'With Stopwords: ', count_stop.most_common(10)
    print 'Single: ', count_single.most_common(10)
    print 'Hash: ', count_hash.most_common(10)
    print 'Terms: ', count_terms.most_common(10)
    print 'Bigrams: ', count_bigrams.most_common(10)


if __name__ == "__main__":
    polarity_distribution()
    
    #distributuion of polarity
    #    tweets.sentiment_type.value_counts().plot(kind='bar',title="Sentiment Analysis")
    
    
    #Graph which shows Compound polarity by hour
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    df=(tweets.groupby('hour',as_index=False).sentiment_compound_polarity.mean())
    print(ggplot(aes(x='hour',y='sentiment_compound_polarity'),data=df)+geom_line())
    
    print 'Term Occurence:'
    term_occurence()
#    print 'WordCloud--'
#    tweets = process_data(tweets)
#    wordcloud_by_province(tweets) 