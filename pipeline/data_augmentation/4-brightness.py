#!/usr/bin/env python3
"""4-brightness.py"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Change Brightness Aug"""

    return tf.image.random_brightness(
        image, max_delta
    )
