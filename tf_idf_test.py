from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.", "The dog.", "The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document. There are 3 sentences in text array. Return 3 vectors.
for i,_ in enumerate(text):
  vector = vectorizer.transform([text[0]])
  # summarize encoded vector
  print(vector.shape)
  print(vector.toarray())

