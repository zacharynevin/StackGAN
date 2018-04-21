import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope

class StackGANGenerator():
    def __init__(self, data_format):
        """
        Initialize StackGAN++ generator

        Params:
            data_format  (str): The data format to use for the image.
        """
        self.data_format = data_format

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
                    output       = slim.fully_connected(z, 4*4*64, activation_fn=None)
                    output_shape = [-1, 4, 4, 64] if self.data_format == 'NHWC' else [-1, 64, 4, 4]
                    output       = slim.batch_norm(output)
                    output       = tf.nn.leaky_relu(output)
                    output       = tf.reshape(output, output_shape)

                G0, output = self.G0(output)
                G1, output = self.G1(output, z)
                G2, output = self.G2(output, z)

                return G0, G1, G2, G_scope

    def G0(self, output):
        with tf.variable_scope('G0'):
            output = self.upscale(output, [-1, 8, 8, 32])
            output = self.upscale(output, [-1, 16, 16, 16])
            output = self.upscale(output, [-1, 32, 32, 8])
            output = self.upscale(output, [-1, 64, 64, 4])
            G0     = tf.nn.tanh(self.upscale(output, [-1, 64, 64, 3], leaky_relu=False, batch_norm=False))
            return G0, output

    def G1(self, output, z):
        with tf.variable_scope('G1'):
            output = self.joint_conv(output, z, 64)
            output = self.residual_block(output, 64)
            output = self.residual_block(output, 64)
            output = self.upscale(output, [-1, 128, 128, 2])
            G1     = tf.nn.tanh(self.upscale(output, [-1, 128, 128, 3], leaky_relu=False, batch_norm=False))
            return G1, output

    def G2(self, output, z):
        with tf.variable_scope('G2'):
            output = self.joint_conv(output, z, 32)
            output = self.residual_block(output, 32)
            output = self.residual_block(output, 32)
            output = self.upscale(output, [-1, 256, 256, 1])
            G2     = tf.nn.tanh(self.upscale(output, [-1, 256, 256, 3], leaky_relu=False, batch_norm=False))
            return G2, output

    def joint_conv(self, input_tensor, z, filters):
        with tf.name_scope('joint_conv'):
            input_shape = input_tensor.get_shape().as_list()

            if self.data_format == 'NCHW':
                channels = input_shape[1]
                height   = input_shape[2]
                width    = input_shape[3]

                z = tf.expand_dims(z, -1)
                z = tf.expand_dims(z, -1)
                z = tf.tile(z, [1, 1, height, width])
                output = tf.concat([input_tensor, z], 1)
            else:
                height   = input_shape[1]
                width    = input_shape[2]
                channels = input_shape[3]

                z = tf.expand_dims(z, 1)
                z = tf.expand_dims(z, 1)
                z = tf.tile(z, [1, height, width, 1])
                output = tf.concat([input_tensor, z], -1)

            output = slim.conv2d(output, filters, kernel_size=3, stride=1, padding='same')
            output = slim.batch_norm(output)
            output = tf.nn.leaky_relu(output)

            return output

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
                 input_tensor,
                 output_shape,
                 kernel_size=3,
                 stride=1,
                 batch_norm=True,
                 leaky_relu=True):

        filters = output_shape[-1]
        height  = output_shape[1]
        width   = output_shape[2]

        with tf.name_scope('upsample_%d_%d_%d' % (height, width, filters)):
            output = self.resize(input_tensor, [height, width])

            output = slim.conv2d(output, filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding='same')

            if batch_norm:
                output = slim.batch_norm(output)
            if leaky_relu:
                output = tf.nn.leaky_relu(output)
            return output

    def resize(self, input_tensor, dims):
        height = dims[0]
        width  = dims[1]

        input_tensor_shape = input_tensor.get_shape().as_list()

        if self.data_format == 'NCHW':
            input_tensor_height = input_tensor_shape[2]
            input_tensor_width = input_tensor_shape[3]
        else:
            input_tensor_height = input_tensor_shape[1]
            input_tensor_width = input_tensor_shape[2]

        is_same_height_width = input_tensor_height == height and input_tensor_width == width

        if is_same_height_width:
            return input_tensor

        with tf.name_scope('resize'):
            if self.data_format == 'NCHW':
                input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])

            output  = tf.image.resize_nearest_neighbor(
                input_tensor,
                (height, width)
            )

            if self.data_format == 'NCHW':
                output = tf.transpose(output, [0, 3, 1, 2])

            return output
