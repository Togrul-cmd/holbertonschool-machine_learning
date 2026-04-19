#!/usr/bin/env python3
"""
ResNet-50 implementation
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds ResNet-50 architecture
    """

    he_init = K.initializers.he_normal(seed=0)

    X = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2),
        padding='same',
        kernel_initializer=he_init
    )(X)

    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)

    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same'
    )(x)

    x = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=he_init
    )(x)

    return K.Model(inputs=X, outputs=x)
