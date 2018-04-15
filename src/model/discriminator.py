import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class StackGANDiscriminator():
    def __init__(self, num_classes, data_format):
        """
        Initialize StackGAN++ Discriminator

        Params:
            num_classes  (int): The number of classes.
            data_format  (str): The data format to use for the image.
        """
        self.num_classes = num_classes
        self.data_format = data_format

    def __call__(self, Im0, Im1, Im2, labels):
        """
        Build StackGAN++ Discriminator graph.

        Params:
            input_tensor (Tensor[NCHW | NHWC]). An image tensor tensor representing the class label.

        Returns:
            Tensor[None]: A 1-D tensor representing the probability (between 0 and 1) that each image in the batch is real.
            Tensor[None, num_classes]: A tensor representing the classification for each image in the batch.
        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
                with tf.variable_scope('D0'):
                    with tf.variable_scope('downsample'):
                        # 64x64x3 -> 4x4x8
                        output = self.encode_x16(Im0)
                        output = slim.conv2d(output, 8, kernel_size=3, stride=1, padding='same')
                        output = slim.batch_norm(output)
                        output = tf.nn.leaky_relu(output)
                        output = slim.flatten(output)

                    D0_uncond = self.uncond(output)
                    D0_cond   = self.cond(output, labels)

                with tf.variable_scope('D1'):
                    with tf.variable_scope('downsample'):
                        # 128x128x3 -> 8x8x512
                        output = self.encode_x16(Im1)
                        # 8x8x512 -> 4x4x8
                        output = slim.conv2d(output, 8, kernel_size=3, stride=2, padding='same')
                        output = slim.batch_norm(output)
                        output = tf.nn.leaky_relu(output)
                        output = slim.flatten(output)

                    D1_uncond = self.uncond(output)
                    D1_cond   = self.cond(output, labels)

                with tf.variable_scope('D2'):
                    with tf.variable_scope('downsample'):
                        # 256x256x3 -> 16x16x512
                        output = self.encode_x16(Im2)
                        # 16x16x512 -> 8x8x16
                        output = slim.conv2d(output, 16, kernel_size=3, stride=2, padding='same')
                        output = slim.batch_norm(output)
                        output = tf.nn.leaky_relu(output)
                        output = slim.conv2d(output, 8, kernel_size=3, stride=2, padding='same')
                        output = slim.batch_norm(output)
                        output = tf.nn.leaky_relu(output)
                        output = slim.flatten(output)

                    D2_uncond = self.uncond(output)
                    D2_cond   = self.cond(output, labels)

        return D0_uncond, D0_cond, D1_uncond, D1_cond, D2_uncond, D2_cond

    def uncond(self, input_tensor):
        with tf.variable_scope('uncond'):
            output = slim.fully_connected(input_tensor, 1)
            return tf.nn.sigmoid(output)

    def cond(self, input_tensor, labels):
        with tf.variable_scope('cond'):
            output = tf.concat([input_tensor, labels], -1)
            output   = slim.fully_connected(output, self.num_classes)
            return tf.nn.sigmoid(output)

    def encode_x16(self, input_tensor):
        with tf.name_scope('encode_x16'):
            with slim.arg_scope([slim.conv2d], kernel_size=4, stride=2, padding='same'):
                output = slim.conv2d(input_tensor, 64)
                output = tf.nn.leaky_relu(output)

                output = slim.conv2d(output, 128)
                output = slim.batch_norm(output)
                output = tf.nn.leaky_relu(output)

                output = slim.conv2d(output, 256)
                output = slim.batch_norm(output)
                output = tf.nn.leaky_relu(output)

                output = slim.conv2d(output, 512)
                output = slim.batch_norm(output)
                output = tf.nn.leaky_relu(output)

                return output
