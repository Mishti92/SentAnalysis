from methods import *

path_to_data = 'data/'

def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
#    print test_neg
    new_tweets = load_tweets(path_to_data)
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec, new_tweets_vec =feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg, new_tweets)
    
    nb_model, lr_model= build_models_NLP(train_pos_vec, train_neg_vec)
    
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)
    #Sentiment Analysis for Demonetization Data
    print "Analysis for Demonetization twitter Data"
    predict_values(lr_model,new_tweets_vec)

if __name__ == "__main__":
    main()
