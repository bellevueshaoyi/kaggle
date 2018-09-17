from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.", "The dog.", "The fox"]
# create the transform
# ngram_range=(1,2) means considering word and 2 words
vectorizer = TfidfVectorizer(ngram_range=(1,2))
# tokenize and build vocab
vectorizer.fit(text)

# It will output: {'fox': 3, 'the':7, ...} where 'fox' is the word, and 3 is the index of the word.
# Full output:
# {'the': 13, 'quick': 11, 'brown': 0, 'fox': 3, 'jumped': 5, 'over': 9, 'lazy': 7, 'dog': 2, 'the quick': 17, 
# 'quick brown': 12, 'brown fox': 1, 'fox jumped': 4, 'jumped over': 6, 'over the': 10, 
# 'the lazy': 16, 'lazy dog': 8, 'the dog': 14, 'the fox': 15}
print(vectorizer.vocabulary_)


# Will output:(each item is the inverse document freq for each feature. The first value 1.69xxx maps to
# feature index=0 which is brown.
# [1.69314718 1.69314718 1.28768207 1.28768207 1.69314718 1.69314718
# 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718 1.69314718
# 1.69314718 1.         1.69314718 1.69314718 1.69314718 1.69314718]
print(vectorizer.idf_)

# encode document. There are 3 sentences in text array. Return 3 vectors.
for i,_ in enumerate(text):
  vector = vectorizer.transform([text[0]])
  # summarize encoded vector
  print(vector.shape)
  print(vector.toarray())

