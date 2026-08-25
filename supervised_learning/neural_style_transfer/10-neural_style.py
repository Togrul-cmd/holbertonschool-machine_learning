#!/usr/bin/env python3
"""Neural Style Transfer Implementation."""
import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initialize class object."""
        valid = (isinstance(style_image, np.ndarray)
                 and style_image.ndim == 3 and style_image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        valid_img = (isinstance(content_image, np.ndarray)
                     and content_image.ndim == 3
                     and content_image.shape[2] == 3)
        if not valid_img:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3)
        containing the image to be scaled
        Returns: the scaled image
        """
        valid = (isinstance(image, np.ndarray)
                 and image.ndim == 3 and image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_size = (int(h * scale), int(w * scale))

        image_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

        resized_image = tf.image.resize(
            image_tensor,
            new_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """Create the model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        pooling_layers = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        vgg.save("base_vgg")

        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=pooling_layers)

        vgg.trainable = False

        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a given layer output."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        input_layer = tf.reshape(input_layer, [-1, input_layer.shape[-1]])

        gram = tf.matmul(input_layer, input_layer, transpose_a=True)

        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(tf.shape(input_layer)[0], tf.float32)

    def generate_features(self):
        """Extract the features used to calculate neural style cost."""
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)

        style_outputs = self.model(style_image)
        self.gram_style_features = [self.gram_matrix(style_feature)
                                    for style_feature in style_outputs[:-1]]

        self.content_feature = self.model(content_image)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single layer."""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        ch = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")
        if len(gram_target.shape) != 3 or gram_target.shape != [1, ch, ch]:
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {ch}, {ch}]")

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculate the style cost for all layers."""
        L = len(self.style_layers)
        if not isinstance(style_outputs, list):
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        if len(style_outputs) != L:
            raise TypeError(
                f"style_outputs must be a list with a length of {L}")
        style_cost = 0
        for target, output in zip(self.gram_style_features, style_outputs):
            style_cost += self.layer_style_cost(output, target)
        return style_cost / L

    def content_cost(self, content_output):
        """Calculate the content cost for the generated image."""
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor of shape {s}")
        if content_output.shape != self.content_feature.shape:
            raise TypeError(f"content_output must be a tensor of shape {s}")
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    @staticmethod
    def variational_cost(generated_image):
        """Calculate the variational cost for the generated image."""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError("image must be a tensor of rank 3 or 4")
        if len(generated_image.shape) == 4:
            pixel_dif1 = (generated_image[:, 1:, :, :] -
                          generated_image[:, :-1, :, :])
            pixel_dif2 = (generated_image[:, :, 1:, :] -
                          generated_image[:, :, :-1, :])
        elif len(generated_image.shape) == 3:
            pixel_dif1 = (generated_image[1:, :, :] -
                          generated_image[:-1, :, :])
            pixel_dif2 = (generated_image[:, 1:, :] -
                          generated_image[:, :-1, :])
        else:
            raise TypeError("image must be a tensor of rank 3 or 4")
        return (tf.reduce_sum(tf.abs(pixel_dif1)) +
                tf.reduce_sum(tf.abs(pixel_dif2)))

    def total_cost(self, generated_image):
        """Calculate the total cost for the generated image."""
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor of shape {s}")
        if generated_image.shape != s:
            raise TypeError(f"generated_image must be a tensor of shape {s}")

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)

        outputs = self.model(preprocessed)

        J_style = self.style_cost(list(outputs[:-1]))
        J_content = self.content_cost(outputs[-1])
        J_var = self.variational_cost(generated_image)

        J = self.alpha * J_content + self.beta * J_style + self.var * J_var

        return J, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate the neural style transferred image."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # Initialize generated image as content image
        generated_image = tf.Variable(self.content_image, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            with tf.GradientTape() as tape:
                J, J_content, J_style, J_var = self.total_cost(
                    generated_image)

            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J}, content {J_content},"
                      f" style {J_style}, var {J_var}")

            if float(J) < best_cost:
                best_cost = float(J)
                best_image = generated_image.numpy()

            if i < iterations:
                grads = tape.gradient(J, generated_image)
                optimizer.apply_gradients([(grads, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0))

        return best_image[0], best_cost
