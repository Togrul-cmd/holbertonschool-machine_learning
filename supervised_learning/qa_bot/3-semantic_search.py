#!/usr/bin/env python3
"""Module for Question Answering using a pre-trained BERT model."""

import os
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents to find the most similar text.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
        sentence (str): The sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to the sentence.
    """
    # Load the Universal Sentence Encoder model
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    documents = []

    # Read all markdown files from the provided path
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(os.path.join(corpus_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())

    corpus = [sentence] + documents

    embeddings = model(corpus)

    sentence_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]

    correlations = np.inner(sentence_embedding, doc_embeddings)

    most_similar_idx = np.argmax(correlations)

    return documents[most_similar_idx]
