#!/usr/bin/env python3
'''kjnkn'''
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''sdvsfs'''
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = X.toarray()
    return embeddings, features
