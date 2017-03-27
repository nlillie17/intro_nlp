# coding: utf-8
# Nick lille with Max and Austin
# Problem 4:  machine-learning and NLP
#
# Movie-review sentiment analysis
#
# This problem asks you to model whether a movie review 
# is positive or negative, based on its language-level
# features. 
#

"""
###########
#
# Note: for the example and problem below, you will need
#       three of the "language corpora" (large sourcetexts)
#       from NLTK. To make sure you've downloaded them,
#       run the following:
#
# In[1]: import nltk
# In[2]: nltk.download()
#
# This will open a window. From there, click on the Corpora tab, 
# and then double-click to download and install these three corpora:
#
# names
# movie_reviews
# opinion_lexicon
# 
###########
"""

## Import all of the libraries and data that we will need.
import pandas as pd
import nltk
from nltk.corpus import names  # see the note on installing corpora, above
from nltk.corpus import opinion_lexicon
from nltk.corpus import movie_reviews
from textblob import TextBlob

import random
import math

from sklearn.feature_extraction import DictVectorizer
import sklearn
import sklearn.tree
from sklearn.metrics import confusion_matrix

#
# experiments showing how the feature vectorizer in scikit-learn works...
#

def read_word2vec_model():  
    """ a function that reads a word2vec model from the file
        "word2vec_model.txt" and returns a model object that
        we will usually name m or model...
    """
    file_name = "word2vec_model.txt"
    from gensim.models.word2vec import Word2Vec
    m = Word2Vec.load_word2vec_format(file_name, binary=False)
    print("The model built is", m)
    ## The above line should print
    ## Word2Vec(vocab=43981, size=300, alpha=0.025)
    ## which tells us that the model represents 43981 words with 300-dimensional vectors
    ## The "alpha" is a model-building parameter called the "learning rate."
    ##   Once the model is built, it can't be changed without rebuilding it; we'll leave it.  
    return m

TRY_FEATURE_VECTORIZER = False
if TRY_FEATURE_VECTORIZER == True:
    # converts from dictionaries to feature arrays (vectors)
    v = DictVectorizer(sparse=False)
    FEATS = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    X = v.fit_transform(FEATS)

    print("FEATS are\n", FEATS, "\n")
    print("v is", v)
    print("X is\n", X, "\n")
    print("Forward: ", v.transform({'foo': 4, 'unseen_feature': 3}), "\n")
    print("Inverse: ", v.inverse_transform(X),"\n")



#
# Language-modeling example:  is a name female or male?
#

# a boolean to turn on/off the name-classifier portion of the code...
RUN_NAME_CLASSIFIER = True
if RUN_NAME_CLASSIFIER == True:

    ## Read all of the names in from the nltk corpus and organize them with labels
    male_names = [(name,'m') for name in names.words('male.txt')]
    female_names = [(name,'f') for name in names.words('female.txt')]
    labeled_names = male_names + female_names

    ## Shuffle all of the labeled names
    # random.seed(0)  # RNG seed: use this for "reproduceable random numbers"
    random.shuffle(labeled_names)



    ## Define the feature function; we'll modify this to improve
    ## the classifier's performance. 
    #
    def gender_features(word):
        """ feature function for the female/male names example
            This function should return a dictionary of features,
            which can really be anything we might compute from the input word...
        """
        return {'last_letter': word[-1].lower()}


    ## Compute features and extract labels and names for the whole dataset
    features = [gender_features(name) for (name, gender) in labeled_names]
    labels = [gender for (name, gender) in labeled_names]
    names = [name for (name, gender) in labeled_names]


    ## Change the dictionary of features into an array with DictVectorizer
    ## then, create our vector of input features, X  (our usual name for it...)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)


    ## Split the input features into train, devtest, and test sets
    X_test = X[:500,:]
    X_devtest = X[500:1500,:]
    X_train = X[1500:,:]

    ## Split the output (labels) into train, devtest, and test sets
    Y_test = labels[:500]
    Y_devtest = labels[500:1500]
    Y_train = labels[1500:]

    ## Split the names themselves into train, devtest, and test sets (for reference)
    names_test = names[:500]
    names_devtest = names[500:1500]
    names_train = names[1500:]



    ############
    #
    # All of the set-up for the name-classification task is now ready...
    #
    ############

    ## Train a decision tree classifier
    #
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train, Y_train)  # fit with the training data!

    ## Evaluate on the devtest set (with known labels) and report the accuracy
    print("Score on devtest set: ", dt.score(X_devtest, Y_devtest))

    ## Predict the results for the devtest set and show the confusion matrix.
    Y_guess = dt.predict(X_devtest)
    CM = confusion_matrix(Y_guess, Y_devtest,  labels=['f','m'])
    print("Confusion Matrix:\n", CM)

    ## a function to predict individual names...
    def classify( name, model, feature_vectorizer, feature_function ):
        """ predictor! """
        features = feature_function(name)
        X = feature_vectorizer.transform(features)
        guess = model.predict(X)[0]
        return guess

    # Example to try out...
    LoN = [ "Zach", "Julie", "Colleen", "Melissa", "Ran", "Geoff", "Bob", "Jessica" ]
    # for name in LoN:
    #     guess = classify( name, dt, v, gender_features )
    #     print(guess,name)


    ## Get a list of errors to examine more closely.
    errors = []
    for i in range(len(names_devtest)):
        this_name = names_devtest[i]
        this_features = X_devtest[i:i+1,:]   # slice of all features for name i
        this_label = Y_devtest[i]
        guess = dt.predict(this_features)[0] # predict (guess) from the features...
        #
        # if the guess was incorrect (wrong label), remember it!
        #
        if guess != this_label:
            errors.append((this_label, guess, this_name))

            
    # Now, print out the results: the incorrect guesses
    # Create a flag to turn this printing on/off...
    # 
    PRINT_ERRORS = True
    if PRINT_ERRORS == True:
        SE = sorted(errors)
        print("There were", len(SE), "errors:")
        print('Name: guess (actual)')
        num_to_print = 10
        for (actual, guess, name) in SE:
            if actual == 'm' and guess == 'f': # adjust these as needed...
                print('  {0:>10}: {1} ({2})'.format(name, guess, actual))
                num_to_print -= 1
                if num_to_print == 0: break
        print()


    ## Finally, score on the test set:
    print("Score on test set: ", dt.score(X_test, Y_test))

    ## Don't actually tune the algorithm for the test set, however!

    ## Reflection / Analysis for the name-classification example
    # 

    """
    Try different features, e.g.,
    + other letters
    + vowel/nonvowel in certain positions
    + presence or absence of one (or more)-letter strings ...
    """






#####################
#
## Problem 4: Movie Review Sentiment starter code...
#
#####################

# a boolean to turn on/off the movie-review-sentiment portion of the code...
RUN_MOVIEREVIEW_CLASSIFIER = True
if RUN_MOVIEREVIEW_CLASSIFIER == True:

    ## Read all of the opinion words in from the nltk corpus.
    #
    pos=list(opinion_lexicon.words('positive-words.txt'))
    neg=list(opinion_lexicon.words('negative-words.txt'))

    ## Store them as a set (it'll make our feature extractor faster).
    # 
    pos_set = set(pos)
    neg_set = set(neg)



    ## Read all of the fileids in from the nltk corpus and shuffle them.
    #
    pos_ids = [(fileid, "pos") for fileid in movie_reviews.fileids('pos')]
    neg_ids = [(fileid, "neg") for fileid in movie_reviews.fileids('neg')]
    labeled_fileids = pos_ids + neg_ids

    ## Here, we "seed" the random number generator with 0 so that we'll all 
    ## get the same split, which will make it easier to compare results.
    #random.seed(0)   # we'll use the seed for reproduceability... 
    random.shuffle(labeled_fileids)



    ## Define the feature function
    #  Problem 4's central challeng is to modify this to improve your classifier's performance...
    #
    def opinion_features(fileid):
        """ feature engineering for movie reviews, dictionary that counts number of positive and negative words in the review. 
            Using, textblob we also check the polarization of words (different gradients of sentiments) and the subjectivity of words 
            (more subjective probably means a more biased review);
            also we tried our hand at doing something similar to context analysis where we check the word before
            the current word and adjust weightings based on the pattern of words (+,-; +,+; -,-; -,+)
    
        """
        # many features are counts!
        positive_count=0.0
        negative_count=0.0
        similar_word_phrase_weight = 2
        prev_word = ''
        for word in movie_reviews.words(fileid):
            w = TextBlob(word)
            if word in pos_set:
                if prev_word in pos_set:
                    positive_count += (1+w.sentiment.polarity*w.sentiment.subjectivity) * similar_word_phrase_weight
                else:
                    positive_count += 1+w.sentiment.polarity*w.sentiment.subjectivity
                   
            elif word in neg_set:
                if prev_word in neg_set:
                    negative_count += (w.sentiment.polarity*w.sentiment.subjectivity-1) * similar_word_phrase_weight
                else:
                    negative_count += w.sentiment.polarity*w.sentiment.subjectivity-1

            prev_word = word


        # here is the dictionary of features...
        features = {'positive': positive_count, 'negative':negative_count}
        return features

    ## Ideas
    # count both positive and negative words...
    # is the ABSOUTE count what matters?
    # 
    # other ideas:
    #
    # feature ideas from the TextBlob library:
    #   * part-of-speech, average sentence length, sentiment score, subjectivity...
    # feature ideas from TextBlob or NLTK (or just Python):
    # average word length
    # number of parentheses in review
    # number of certain punctuation marks in review
    # number of words in review
    # words near or next-to positive or negative words: "not excellent" ?
    # uniqueness
    #
    # many others are possible...


    ## Extract features for all of the movie reviews
    # 
    end=""
    flush=True
    print ("Creating features for all reviews...")
    features = [opinion_features(fileid) for (fileid, opinion) in labeled_fileids]
    labels = [opinion for (fileid, opinion) in labeled_fileids]
    fileids = [fileid for (fileid, opinion) in labeled_fileids]
    print(" ... done.")


    ## Change the dictionary of features into an array
    #
    print("Transforming from dictionaries of features to vectors...")
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(features)
    print(" ... done.")

    ## Split the data into train, devtest, and test

    X_test = X[:100,:]
    Y_test = labels[:100]
    fileids_test = fileids[:100]

    X_devtest = X[100:200,:]
    Y_devtest = labels[100:200]
    fileids_devtest = fileids[100:200]

    X_train = X[200:,:]
    Y_train = labels[200:]
    fileids_train = fileids[200:]

    ## Train the decision tree classifier - perhaps try others or add parameters
    #
    dt = sklearn.tree.DecisionTreeClassifier()
    dt.fit(X_train,Y_train)

    ## Evaluate on the devtest set; report the accuracy and also
    ## show the confusion matrix.
    #
    print("Score on devtest set: ", dt.score(X_devtest, Y_devtest))
    Y_guess = dt.predict(X_devtest)
    CM = confusion_matrix(Y_guess, Y_devtest)
    #print("Confusion Matrix:\n", CM)

    ## Get a list of errors to examine more closely.
    #
    errors = []

    for i in range(len(fileids_devtest)):
        this_fileid = fileids_devtest[i]
        this_features = X_devtest[i:i+1,:]
        this_label = Y_devtest[i]
        guess = dt.predict(this_features)[0]
        if guess != this_label:
            errors.append((this_label, guess, this_fileid))
    headers = ["Actual","Guess","File ID"]
    df = pd.DataFrame(errors, columns= headers)

    print (df)

    # Now, print out the results: the incorrect guesses
    # Create a flag to turn this printing on/off...
    # 
    PRINT_ERRORS = True
    if PRINT_ERRORS == True:
        SE = sorted(errors)
        print("There were", len(SE), "errors")
       
    ''' The reason for the df was for an easy way to see which files the errors were in '''

    ## TODO: Generate a "friendly" display of the errors so that you can use them
    ## to improve the feature extraction function.
    #    This could be by printing (similar to the names example)
    #    or, even by graphing some of the numeric values of features...



    # ## Problem 4 Reflections/Analysis
    #
    # Include a short summary of
    #   (a) how well your final set of features did!

    """ Final feature set ranged from 58% to 71% (!!!) """

    #   (b) what other features you tried and which ones seemed to 
    #       help the most/least

    """ So we used two features of TextBlob: the polarity (how positive or how negative the word is)
    and subjectivity (how subjective the word is). I multiplied the two of them together to make them closer to zero
    (but not exactly zero), and then because negative words should have negative values, I subtracted 1 from the
    negative_count value and added 1 to the positive count values to give a true weighted score.

    After doing hw0, we realized that just looking at the sentiment score wouldn't be sufficient so we tried to do
    some basic context analysis by looking at the previous word. Our hypothesis is that if a string of negative or
    positive words occur then the review is more likely to be negative or positive. It also takes into account emotion.
    Thus, we added a weighting to our positive and negative count based on the pattern of words.

    Something that didn't work for great was features for word length. We thought that this could impact the scores
    but we were wrong. The scores were actually worse than before. """




