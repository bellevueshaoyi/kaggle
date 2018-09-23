# See https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle.
# See http://scikit-learn.org/stable/modules/decomposition.html#lsa for SVD decomposition.
# See https://en.wikipedia.org/wiki/Singular-value_decomposition for PCA vs. SVD.

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import os

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.svm import SVC

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
train.head()

# 1. Preprocessing.
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author)
print(y)

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
print(xtrain.shape)
print(xvalid.shape)

# 2. Get tf-idf.
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

# To print dictionary (all words from training data):
print('number of words in dictionary')
print(len(tfv.vocabulary_))

# 3. Get SVD from tf-idf (get top 250 features instead of using all 15K words)

# 3.1 Apply SVD with 250 components. 
# 120-200 components are good enough for SVM model.
# See http://scikit-learn.org/stable/modules/decomposition.html#lsa.
# When X = U*T*V' (V' is the transpose of V), U is eigenvalue, V is eigenvector, then
# xtrain_svd = xtrain_tfv * V, which is equal to U*T. See 
# http://scikit-learn.org/stable/modules/decomposition.html#lsa.
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# 3.2 Scale the SVD data to mean = 0, standard dev = 1.
# SVD data is eigenvector * scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)
# Verify xtrain_svd_scl is scaled to mean = 0, std = 1.
print(np.min(xtrain_svd_scl))
print(np.max(xtrain_svd_scl))
print(np.median(xtrain_svd_scl))
print(np.std(xtrain_svd_scl))

# 4. Fitting a simple SVM
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)
# Print log loss.
print ("logloss: %0.3f " % log_loss(yvalid, predictions))
