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
        self.Nd = 64

    def D0(self, Im0, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D0', reuse=tf.AUTO_REUSE) as D0_scope:
                net = self.encode_x16(Im0)

                logits = self.logits(net)

                return logits, D0_scope

    def D1(self, Im1, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D1', reuse=tf.AUTO_REUSE) as D1_scope:
                net = self.encode_x16(Im1)
                net = self.downsample(net, 16*self.Nd)
                net = self.conv3x3(net, 8*self.Nd)

                logits = self.logits(net)

                return logits, D1_scope

    def D2(self, Im2, scope=None):
        with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
            with tf.variable_scope(scope or 'discriminator/D2', reuse=tf.AUTO_REUSE) as D2_scope:
                net = self.encode_x16(Im2)
                net = self.downsample(net, 16*self.Nd)
                net = self.downsample(net, 32*self.Nd)
                net = self.conv3x3(net, 16*self.Nd)
                net = self.conv3x3(net, 32*self.Nd)

                logits = self.logits(net)

                return logits, D2_scope

    def conv3x3(self, net, filters):
        with tf.name_scope('conv3x3_block'):
            net = slim.conv2d(net, filters, kernel_size=3, stride=1, padding='same')
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)
            return net

    def downsample(self, net, filters):
        with tf.name_scope('downsample'):
            net = slim.conv2d(net, filters, kernel_size=4, stride=2, padding='same', biases_initializer=None)
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)

            return net

    def logits(self, net):
        with tf.name_scope('logits'):
            net = slim.conv2d(net, 1, kernel_size=4, stride=4, padding='same')
            net = tf.nn.sigmoid(net)
            return tf.reshape(net, [-1])

    def add_noise(self, net):
        with tf.name_scope('noise'):
            noise  = tf.random_normal(tf.shape(net), stddev=0.02, dtype=tf.float32)
            net = net + noise
            return net

    def encode_x16(self, net):
        with tf.name_scope('encode_x16'):
            with slim.arg_scope([slim.conv2d], kernel_size=4, stride=2, padding='same'):
                net = self.add_noise(net)

                net = slim.conv2d(net, self.Nd)
                net = tf.nn.leaky_relu(net)

                net = slim.conv2d(net, 2*self.Nd)
                net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)

                net = slim.conv2d(net, 4*self.Nd)
                net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)

                net = slim.conv2d(net, 8*self.Nd)
                net = slim.batch_norm(net)
                net = tf.nn.leaky_relu(net)

                return net
