import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class StackGANDiscriminator():
    def __init__(self, data_format):
        """
        Initialize StackGAN++ Discriminator

        Params:
            data_format  (str): The data format to use for the image.
        """
        self.data_format = data_format

    def D0(self, Im0, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D0', reuse=tf.AUTO_REUSE) as D0_scope:
                output = self.encode_x16(Im0)

                logits = self.logits(output)

                return logits, D0_scope

    def D1(self, Im1, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D1', reuse=tf.AUTO_REUSE) as D1_scope:
                output = self.encode_x16(Im1)
                output = self.downsample(output, 16)
                output = self.conv3x3(output, 8)

                logits = self.logits(output)

                return logits, D1_scope

    def D2(self, Im2, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D2', reuse=tf.AUTO_REUSE) as D2_scope:
                output = self.encode_x16(Im2)
                output = self.downsample(output, 16)
                output = self.downsample(output, 32)
                output = self.conv3x3(output, 16)
                output = self.conv3x3(output, 8)

                logits = self.logits(output)

                return logits, D2_scope

    def conv3x3(self, input_tensor, filters):
        with tf.name_scope('conv3x3_block'):
            output = slim.conv2d(input_tensor, filters, kernel_size=3, stride=1, padding='same')
            output = slim.batch_norm(output)
            output = tf.nn.leaky_relu(output)
            return output

    def downsample(self, input_tensor, filters):
        with tf.name_scope('downsample'):
            output = slim.conv2d(input_tensor, filters, kernel_size=4, stride=2, padding='same', biases_initializer=None)
            output = slim.batch_norm(output)
            output = tf.nn.leaky_relu(output)

            return output

    def logits(self, input_tensor):
        with tf.variable_scope('logits'):
            output = slim.conv2d(input_tensor, 1, kernel_size=4, stride=4, padding='same')
            output = tf.nn.sigmoid(output)
            return tf.reshape(output, [-1])

    def add_noise(self, output):
        with tf.name_scope('noise'):
            noise  = tf.random_normal(tf.shape(output), stddev=0.02, dtype=tf.float32)
            output = output + noise
            return output

    def encode_x16(self, output, add_noise=False):
        with tf.name_scope('encode_x16'):
            with slim.arg_scope([slim.conv2d], kernel_size=4, stride=2, padding='same'):
                output = self.add_noise(output)

                output = slim.conv2d(output, 64)
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
