#!/usr/bin/env python3
"""1-crop.py"""
import tensorflow as tf


def crop_image(image, size):
    """Cropping aug"""

    return tf.image.random_crop(image, size)
