#!/usr/bin/env python3
"""Module for Question Answering using a pre-trained BERT model."""

qa_function = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts using a loop.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input("Q: ")

        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break

        # Step 1: Perform semantic search to find the most relevant document
        reference = semantic_search(corpus_path, user_input)

        # Step 2: Extract the exact answer using the BERT QA model
        answer = qa_function(user_input, reference)

        # Step 3: Handle edge cases where no valid answer is found
        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
