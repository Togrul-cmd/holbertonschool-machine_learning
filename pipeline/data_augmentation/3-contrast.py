#!/usr/bin/env python3
"""3-contrast.py"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Contrast Change Aug"""

    return tf.image.random_contrast(
        image,
        lower,
        upper
    )
