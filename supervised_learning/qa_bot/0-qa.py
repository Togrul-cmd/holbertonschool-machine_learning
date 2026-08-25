#!/usr/bin/env python3
"""
Module for Question Answering using a pre-trained BERT model.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): A string containing the question to answer.
        reference (str): A string containing the reference document from which
                         to find the answer.

    Returns:
        str: A string containing the answer.
        None: If no valid answer is found within the reference document.
    """
    # Load the pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Load the QA model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--hub/bert-uncased-tf2-qa/1")

    # Tokenize the question and reference text
    # return_tensors='tf' ensures the output is compatible with TensorFlow
    inputs = tokenizer([question], [reference], return_tensors='tf')

    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Pass the inputs through the model
    # Note: This specific TF Hub model expects a list of these three tensors
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # The model returns a list where index 0 is start_logits and 1 is end_logits
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Get the index with the highest probability for start and end positions
    start_idx = tf.argmax(start_logits, axis=1)[0].numpy()
    end_idx = tf.argmax(end_logits, axis=1)[0].numpy()

    # Determine if a valid answer was found
    # Index 0 is the [CLS] token, which the model uses to indicate "no answer"
    if start_idx == 0 or end_idx == 0 or start_idx > end_idx:
        return None

    # Slice the token IDs to extract the answer
    answer_tokens = input_word_ids[0][start_idx:end_idx + 1]

    # Convert the token IDs back into a string
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer
