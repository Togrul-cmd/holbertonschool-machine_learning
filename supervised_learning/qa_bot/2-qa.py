#!/usr/bin/env python3
"""Module for Question Answering using a pre-trained BERT model."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in an interactive loop.

    Args:
        reference (str): The reference document containing potential answers.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input("Q: ")

        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(user_input, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
