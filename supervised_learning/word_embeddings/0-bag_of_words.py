#!/usr/bin/env python3
'''kjnkn'''
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    '''sdvsfs'''
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = X.toarray()
    return embeddings, features
