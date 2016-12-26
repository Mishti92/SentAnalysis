Project done as a part of Natural Language Processing Course at Mcgill university

Sentiment Analysis is the computational treatment of opinions, sentiments and subjectivity of text. This report will review and evaluate some of the various techniques used for sentiment analysis. Different classification algorithms such as Naive Bayes, Logistic Regression and a artificial neural network technique - Doc2Vec has been implemented and compared on the twitter data set for best performance. Logistic Regression using doc2vec model performs comparatively better. Also, the above techniques and NLTK's Vader will be used to analyze the sentiments of tweets related to aftermath of currency demonetization in India.
 
 
Check the report for further details.
 
File descriptions -
 naivebayes.py : Trains the Sentiment140 tweets dataset on usual naive bayes and logistic regression classification algorithms. 
 If running from terminal , run using python naivebayes_logreg.py
 
 doc2vec.py : Uses Distributed Memory - dm model of Doc2Vec to train the dataset. 
 To run - python doc2vec.py 
 Note - This may take some time to run.
 
 stream_data.py : Used to stream data from twitter to stream folder. 
 
 methods.py : Different functions defined. Imported in other fies.
 
 vader.py : Runs sentiment analysis on the test set and streamed twitter data using NLTK's SentimentIntensityAnalyzer
 
 /output -
 Output files with sentiment Analysis done on tweets using different methods
 
 /data - FOLDER NOT INCLUDED 
 Sentiment 140 data set divided in train-pos, train-neg, test-pos, test-neg files.
    /stream
    json and csv files contaning tweets from live data stream
