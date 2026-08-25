#!/usr/bin/env python3
"""Module for executing an interactive question and answer loop."""

if __name__ == '__main__':
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input("Q: ")

        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")
