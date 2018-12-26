import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter
from itertools import *
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.svm import SVC

def _predict(train, col_name, verbal=True):
    # 1. Preprocessing.
    # TBA
    lbl_enc = preprocessing.LabelEncoder()
    y_kernel = lbl_enc.fit_transform(train.Kernel)
    # 2. Train test split
    x_train, x_test, y_kernel_train, y_kernel_test = train_test_split(train.Description.values, 
                                                                      y_kernel,
                                                                      stratify=y_kernel, test_size=0.1)    
    # 3. Build tf-idf vector.
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
    tfv.fit(list(x_train))
    x_train_tfv =  tfv.transform(x_train) 
    x_test_tfv = tfv.transform(x_test)

    # To print dictionary (all words from training data):
    if verbal:
        print('---number of words in dictionary---')
        print(len(tfv.vocabulary_))
    
    # 4. Analyze tf-idf dictionary.
    # 4.1 Print the stop words. Terms that were ignored because they either: 
    # a) occurred in too many documents (max_df) 
    # b) occurred in too few documents (min_df) 
    # c) were cut off by feature selection (max_features).
    # Uncomment to see stop words.
    if verbal:
        print('---stop words: words that are too frequent or too few---')
        for i, val in enumerate(islice(tfv.stop_words_, 10)):
            print(val)
    
    # 5. Regression
    clf = LogisticRegression(C=1.0, solver='newton-cg')
    clf.fit(x_train_tfv, y_kernel_train)
    predictions = clf.predict_proba(x_test_tfv)

    # Use log-loss: sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred)).
    if verbal:
        print('----log loss---')
        print ("logloss: %0.3f " % log_loss(y_kernel_test, predictions))
    
    # 6. Now apply back to the whole set
    z = tfv.transform(train['Description'].values)
    final_result = clf.predict_proba(z)
    return ['NO' if r[0] > 0.5 else 'Yes' for r in final_result]
    
    
print(os.listdir("../input/"))
path = "../input/vulnerability.csv"
train = pd.read_csv(path)
train.head(5)
output = train

# 'Code Execution' only has one value.
output = train
for col in ['Kernel', 'Remote', 'Privilege Escalation', 'Denial of Service']:
    print('column {0}'.format(col))
    y = _predict(train, col, verbal=False)
    # print('--- Stats: {0}---'.format(Counter(y)))
    predicted_output_col = 'PREDICTED ' + col
    output[predicted_output_col] = y
    
    # Evaluate
    count=0.0
    for _,o in output.iterrows():
        if o[col] == o[predicted_output_col]:
            count = count + 1
    print('precision: {0}'.format(count/len(output)))
    print('=========================')

    
output.head(20)    
    
