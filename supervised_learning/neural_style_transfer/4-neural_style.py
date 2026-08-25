#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
import tensorflow as tf


class NST:
    """A class that does the trick"""
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the class"""
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(alpha, (int, float)) or
                alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (not isinstance(beta, (int, float)) or
                beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Scale the image"""
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
        image_h, image_w, image_c = image.shape
        if image_h <= 0 or image_w <= 0 or image_c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if image_h > image_w:
            new_h = 512
            new_w = int(image_w * 512 / image_h)
        else:
            new_w = 512
            new_h = int(image_h * 512 / image_w)
        resized_image = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )
        rescaled_image = resized_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0.0, 1.0)
        rescaled_image = tf.expand_dims(rescaled_image, 0)
        return rescaled_image

    def load_model(self):
        """Load the model"""
        VGG19_model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')
        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the gram matrix"""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "input_layer must be a tensor of rank 4"
            )
        if len(input_layer.shape) is not 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, image_h, image_w, image_c = input_layer.shape
        product = int(image_h * image_w)
        features = tf.reshape(input_layer, (product, image_c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, 0)
        gram = gram / tf.cast(product, tf.float32)
        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        """
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )

        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost"""
        if not isinstance(style_output,
                          (tf.Tensor,
                           tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, h, w, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                "gram_target must be a tensor of shape "
                f"[1, {c}, {c}]"
            )

        if gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape "
                f"[1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)

        style_cost = tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )
        return style_cost
