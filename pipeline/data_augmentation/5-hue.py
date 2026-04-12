#!/usr/bin/env python3
"""5-hue.py"""
import tensorflow as tf


def change_hue(image, delta):
    """Change Hue Aug"""

    return tf.image.random_hue(
        image, max_delta
    )
