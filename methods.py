import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.svm
import nltk
import re
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec


path_to_data = 'data/'

def cleanhtml(raw_html):
#     cleanr = re.compile('<.*?>')
    cleantext = re.sub(r"(?:\@|https?\://)\S+", "", raw_html)
#     cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]


tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def remove_non_ascii_2(text):
    return re.sub(r'[^\x00-\x7F]+',' ', text)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt','RT','via','PM','Modi','amp','fu']
# print stop


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def load_tweets(path_to_dir):
    new_tweets=[]
    tweets=pd.read_csv(path_to_dir+"tweets.csv",encoding = "ISO-8859-1")
    for i in range(len(tweets['text'])):
        line = tweets['text'][i]
        line = remove_non_ascii_2(line)
#        print line
        line = cleanhtml(line)
#        line = line.decode("utf8").encode('ascii','ignore')
        words =[term for term in preprocess(line) if term not in stop]
#                
#            words = [w.lower() for w in line.split() if len(w)>=3]
        new_tweets.append(words)
    return new_tweets
    
    

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg, new_tweets):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    pos_total_dict={}
    for line in train_pos:
        pos_word_dict={}
        for word in line:
            pos_word_dict[word]=1
        for word in pos_word_dict.keys():
            if word in pos_total_dict:
                pos_total_dict[word]=pos_total_dict[word]+1
            else:
                pos_total_dict[word]=1
    
    neg_total_dict={}
    for line in train_neg:
        neg_word_dict={}
        for word in line:
            neg_word_dict[word]=1
        for word in neg_word_dict.keys():
            if word in neg_total_dict:
                neg_total_dict[word]=neg_total_dict[word]+1
            else:
                neg_total_dict[word]=1
    #print len(pos_total_dict)
    #print len(neg_total_dict)
    pos_without_sw=remove_stopwords(pos_total_dict,stopwords)
    neg_without_sw=remove_stopwords(neg_total_dict,stopwords)
    at_least_one_words_pos=at_least_one(pos_without_sw,train_pos)
    at_least_one_words_neg=at_least_one(neg_without_sw,train_neg)
    final_dict=at_least_twice(at_least_one_words_pos,at_least_one_words_neg,pos_total_dict,neg_total_dict)

    feature_list=final_dict.keys()
    
    train_pos_vec=[]
    train_neg_vec=[]
    test_pos_vec=[]
    test_neg_vec=[]
    new_tweets_vec = []

    for line in train_pos:
        line_dict={}
        for word in line:
            line_dict[word]=1
        vector_list=[]
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        train_pos_vec.append(vector_list)

    for line in train_neg:
        line_dict={}
        for word in line:
            line_dict[word]=1
        vector_list=[]
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        train_neg_vec.append(vector_list)
    
    for line in test_pos:
        line_dict={}
        for word in line:
            line_dict[word]=1
        vector_list=[]
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        test_pos_vec.append(vector_list)
    
    for line in test_neg:
        line_dict={}
        for word in line:
            line_dict[word]=1
        vector_list=[]
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        test_neg_vec.append(vector_list)
        
#    print 'I am here'
#    print new_tweets
    for line in new_tweets:
        line_dict={}
        for word in line:
            line_dict[word]=1
#        print line_dict
        vector_list=[]
        for word in feature_list:
            if word in line_dict:
                vector_list.append(1)
            else:
                vector_list.append(0)
        new_tweets_vec.append(vector_list)
            
    # Return the five feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec, new_tweets_vec



def remove_stopwords(total_dict,stopwords):
    filtereddict={k:v for k,v in total_dict.items() if k not in stopwords}
    return filtereddict

def at_least_one(without_sw,train):
    threshold=len(train)*0.01
    reqlist={k:v for k,v in without_sw.items() if v>=threshold}
    return reqlist

def at_least_twice(words_pos,words_neg,pos_total_dict,neg_total_dict):
    #final_dict_pos={k:v for k,v in words_pos.i
    final_dict={}
    #print len(words_pos)
    #print len(words_neg)
    
    for word in words_pos.keys():
        if not word in neg_total_dict:
            final_dict[word]=1
        elif words_pos[word]>=2*neg_total_dict[word]:
            final_dict[word]=1
    
    #print len(final_dict)     
    for word in words_neg.keys():
        if not word in pos_total_dict:
            final_dict[word]=1
        elif words_neg[word]>=2*pos_total_dict[word]:
            final_dict[word]=1
    return final_dict

def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X=train_pos_vec+train_neg_vec
    logr_model=sklearn.linear_model.LogisticRegression()
    lr_model=logr_model.fit(X,Y)
    
    gnb=sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model=gnb.fit(X,Y)   

    return nb_model, lr_model

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg, new_tweets):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos=[]
    #print train_pos[0:2]
    for i,list_of_words in enumerate(train_pos):
        so=LabeledSentence(words=list_of_words,tags=['train_pos_'+str(i)])
        labeled_train_pos.append(so)

    labeled_train_neg=[]
    for i,list_of_words in enumerate(train_neg):
        labeled_train_neg.append(LabeledSentence(words=list_of_words,tags=["train_neg_"+str(i)]))
    
    labeled_test_pos=[]
    for i,list_of_words in enumerate(test_pos):
        labeled_test_pos.append(LabeledSentence(words=list_of_words,tags=["test_pos_"+str(i)]))
    
    labeled_test_neg=[]
    for i,list_of_words in enumerate(test_neg):
        labeled_test_neg.append(LabeledSentence(words=list_of_words,tags=["test_neg_"+str(i)]))
        
    labeled_new_tweets=[]
    for i,list_of_words in enumerate(new_tweets):
        labeled_new_tweets.append(LabeledSentence(words=list_of_words,tags=["new_tweets_"+str(i)]))
    
#    print labeled_new_tweets[0:3]
#    sys.exit(0)

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg + labeled_new_tweets
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    ##Get training set vectors from our models
    #Use the docvecs function to extract the feature vectors for the training and test data
    train_pos_vec=[]
    for i,line in enumerate(train_pos):
        train_pos_vec.append(model.docvecs['train_pos_'+str(i)])
   
    train_neg_vec=[]
    for i,line in enumerate(train_neg):
        train_neg_vec.append(model.docvecs['train_neg_'+str(i)])
    
    test_pos_vec=[]
    for i,line in enumerate(test_pos):
        test_pos_vec.append(model.docvecs['test_pos_'+str(i)])
    
    test_neg_vec=[]
    for i,line in enumerate(test_neg):
        test_neg_vec.append(model.docvecs['test_neg_'+str(i)])
    
    new_tweets_vec=[]
    for i,line in enumerate(new_tweets):
        new_tweets_vec.append(model.docvecs['new_tweets_'+str(i)])


    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec, new_tweets_vec

def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X=train_pos_vec+train_neg_vec
    logr_model=sklearn.linear_model.LogisticRegression()
    lr_model=logr_model.fit(X,Y)
    
    gnb=sklearn.naive_bayes.GaussianNB()
    nb_model=gnb.fit(X,Y)   

    
    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=True):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    predict1=model.predict(test_pos_vec).tolist()
    predict2=model.predict(test_neg_vec).tolist()
    #print predict1[0:5]
    #print type(predict1) is list
    #sys.exit(0)
    tp=predict1.count('pos')
    fn=predict1.count('neg')
    tn=predict2.count('neg')
    fp=predict2.count('pos')    
    accuracy=float((tp+tn))/(len(test_pos_vec)+len(test_neg_vec))

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)
    
def predict_values(model,new_vec):
    """Predicts sentiments for a new data set.
    """
    
    predict = model.predict(new_vec).tolist()
    print 'positive values', predict.count('pos')
    print 'negative values', predict.count('neg')