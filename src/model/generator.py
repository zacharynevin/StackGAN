import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from src.model.helpers import *

class StackGANGenerator():
    def __init__(self, data_format):
        """
        Initialize StackGAN++ generator

        Params:
            data_format  (str): The data format to use for the image.
        """
        self.data_format = data_format
        self.Ng = 32

    def __call__(self, z):
        """
        Build StackGAN++ generator graph

        Params:
            z Tensor[None, None]: A 2-D tensor representing the z-input.

        Returns:
            G0 Tensor[None, 64, 64, 3]
            G1 Tensor[None, 128, 128, 3]
            G2 Tensor[None, 256, 256, 3]
        """
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as G_scope:
            with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):

                with tf.variable_scope('FC'):
                    net       = slim.fully_connected(z, 4*4*64*self.Ng, activation_fn=None)
                    net_shape = [-1, 4, 4, 64*self.Ng] if self.data_format == 'NHWC' else [-1, 64*self.Ng, 4, 4]
                    net       = slim.batch_norm(net)
                    net       = tf.nn.leaky_relu(net)
                    net       = tf.reshape(net, net_shape)

                G0, net = self.G0(net)
                G1, net = self.G1(net, z)
                G2, net = self.G2(net, z)

                return G0, G1, G2, G_scope

    def G0(self, net):
        with tf.variable_scope('G0'):
            net = self.upscale(net, [-1, 8, 8,   32*self.Ng])
            net = self.upscale(net, [-1, 16, 16, 16*self.Ng])
            net = self.upscale(net, [-1, 32, 32, 8*self.Ng])
            net = self.upscale(net, [-1, 64, 64, 4*self.Ng])
            G0  = tf.nn.tanh(self.upscale(net, [-1, 64, 64, 3], leaky_relu=False, batch_norm=False))
            return G0, net

    def G1(self, net, z):
        with tf.variable_scope('G1'):
            net = self.joint_conv(net, z, 64)
            net = self.residual_block(net, 64)
            net = self.residual_block(net, 64)
            net = self.upscale(net, [-1, 128, 128, 2*self.Ng])
            G1  = tf.nn.tanh(self.upscale(net, [-1, 128, 128, 3], leaky_relu=False, batch_norm=False))
            return G1, net

    def G2(self, net, z):
        with tf.variable_scope('G2'):
            net = self.joint_conv(net, z, 32)
            net = self.residual_block(net, 32)
            net = self.residual_block(net, 32)
            net = self.upscale(net, [-1, 256, 256, 1*self.Ng])
            G2  = tf.nn.tanh(self.upscale(net, [-1, 256, 256, 3], leaky_relu=False, batch_norm=False))
            return G2, net

    def joint_conv(self, net, z, filters):
        with tf.name_scope('joint_conv'):
            net_shape = net.get_shape().as_list()

            if self.data_format == 'NCHW':
                channels = net_shape[1]
                height   = net_shape[2]
                width    = net_shape[3]

                z = tf.expand_dims(z, -1)
                z = tf.expand_dims(z, -1)
                z = tf.tile(z, [1, 1, height, width])
                net = tf.concat([net, z], 1)
            else:
                height   = net_shape[1]
                width    = net_shape[2]
                channels = net_shape[3]

                z = tf.expand_dims(z, 1)
                z = tf.expand_dims(z, 1)
                z = tf.tile(z, [1, height, width, 1])
                net = tf.concat([net, z], -1)

            net = slim.conv2d(net, filters, kernel_size=3, stride=1, padding='same')
            net = slim.batch_norm(net)
            net = tf.nn.leaky_relu(net)

            return net

    def residual_block(self, x, filters):
        with tf.name_scope('residual_block'):
            with slim.arg_scope([slim.conv2d], stride=1, padding='same'):
                Fx  = slim.conv2d(x, filters, kernel_size=3)
                Fx  = slim.batch_norm(Fx)
                Fx  = tf.nn.leaky_relu(Fx)
                Fx  = slim.conv2d(Fx, filters, kernel_size=3)
                Fx  = slim.batch_norm(Fx)

                x   = slim.conv2d(x, filters, kernel_size=1)
                x   = slim.batch_norm(x)

                Fxx = Fx + x
                return tf.nn.leaky_relu(Fxx)

    def upscale(self,
                 net,
                 output_shape,
                 kernel_size=3,
                 stride=1,
                 batch_norm=True,
                 leaky_relu=True):

        filters = output_shape[-1]
        height  = output_shape[1]
        width   = output_shape[2]

        with tf.name_scope('upsample_%d_%d_%d' % (height, width, filters)):
            net = self.resize(net, [height, width])

            net = slim.conv2d(net, filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding='same')

            if batch_norm:
                net = slim.batch_norm(net)
            if leaky_relu:
                net = tf.nn.leaky_relu(net)
            return net

    def resize(self, net, dims):
        height = dims[0]
        width  = dims[1]

        net_shape = net.get_shape().as_list()

        if self.data_format == 'NCHW':
            net_height = net_shape[2]
            net_width = net_shape[3]
        else:
            net_height = net_shape[1]
            net_width = net_shape[2]

        is_same_height_width = net_height == height and net_width == width

        if is_same_height_width:
            return net

        with tf.name_scope('resize'):
            if self.data_format == 'NCHW':
                net = nchw_to_nhwc(net)

            net  = tf.image.resize_nearest_neighbor(net, (height, width))

            if self.data_format == 'NCHW':
                net = nhwc_to_nchw(net)

            return net
