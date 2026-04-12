#!/usr/bin/env python3
"""2-rotate.py"""
import tensorflow as tf


def rotate_image(image):
    """Rotate aug"""

    return tf.image.rot90(image)
