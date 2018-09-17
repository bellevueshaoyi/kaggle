# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

print(os.listdir("../input"))

# 1. Preprocessing.
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author)
# Labels converted from name to int.
print y

# 2. train test split.
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)

#3. Build Tfidf vector

tfv = TfidfVectorizer(
    # min_df means has to at least occur in 3 documents.
    min_df=3, 
    # use all words as features. When ngaram is set, use all ngaram combinations.
    max_features=None, 
    #strip_accents: take unicode chars.
    strip_accents='unicode', 
    # analyzer='word': analyze word. Another option is 'character'.
    analyzer='word',
    # words are defined as \w{1,}, which is reg exp for words with length
    # of 1 or more.  r means this is a raw string.
    token_pattern=r'\w{1,}',
    # ngram_range=(1,3): features are either 1 word, or  2 or 3 consequective words.
    ngram_range=(1, 3), 
    # use_idf=1: use inverse document feature. Words that appear in many 
    # documents will (like 'the', 'a') will have lower score.
    use_idf=1,
    # If a word from vocabulary was never seen in the train data, 
    # but occures in the test, smooth_idf allows it to be successfully 
    # processed. See https://stackoverflow.com/questions/47069744/is-smooth-idf-redundant.
    smooth_idf=1,
    # Replace tf with 1 + log(tf).
    # Where a term that is X times more frequent shouldn't be X times as important. 
    # sublinear_tf causes logarithmic increase in Tfidf score compared to the term frequency. 
    # See https://stackoverflow.com/questions/34435484/tfidfvectorizer-normalisation-bias.
    sublinear_tf=1,
    # automatically detect stop words. Terms that were ignored because they either: 
    # a) occurred in too many documents (max_df) 
    # b) occurred in too few documents (min_df) 
    # c) were cut off by feature selection (max_features).
    stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)

# 4. print the vocabulary that's built by TF-IDF (including the ngaram combinations) and stop words.
# Print the stop words. Terms that were ignored because they either: 
# a) occurred in too many documents (max_df) 
# b) occurred in too few documents (min_df) 
# c) were cut off by feature selection (max_features).
tfv.stop_words_

# Print all the features.
print (tfv.vocabulary_)

# 5. Regression
clf = LogisticRegression(C=1.0, solver='newton-cg')
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

# Use log-loss: sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred)).
print ("logloss: %0.3f " % log_loss(yvalid, predictions))
