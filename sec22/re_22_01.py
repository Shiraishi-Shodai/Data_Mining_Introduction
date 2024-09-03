from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    '吾輩 は ネコ で ある 。',
    '松山 で の ネコ の 暮らし は 楽 で ある 。',
    '楽 なのは よい',
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())