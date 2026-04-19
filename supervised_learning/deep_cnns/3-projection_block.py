#!/usr/bin/env python3
"""Projection block module for ResNet"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block

    Parameters:
    A_prev: output from previous layer
    filters: tuple/list (F11, F3, F12)
    s: stride for first conv in main and shortcut path

    Returns:
    Activated output of projection block
    """
    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=0)

    # Main path
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add + activation
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
