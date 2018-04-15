import math
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class StackGANGenerator():
    def __init__(self, num_classes, data_format):
        """
        Initialize StackGAN++ generator

        Params:
            num_classes  (int). The number of classes
            data_format  (str): The data format to use for the image.
        """
        self.num_classes = num_classes
        self.data_format = data_format

    def __call__(self, z, labels):
        """
        Build StackGAN++ generator graph

        Params:
            z       (Tensor[None, None]): A 2-D tensor representing the z-input.
            labels  (Tensor[None, None]): A 2-D tensor representing the labels for each z-input.

        Returns:
            G0 Tensor[None, 64, 64, 3]
            G1 Tensor[None, 128, 128, 3]
            G2 Tensor[None, 256, 256, 3]
        """
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=self.data_format):
                z = tf.concat([z, labels], 1)

                with tf.variable_scope('FC'):
                    output       = slim.fully_connected(z, 4*4*64*2, activation_fn=None)
                    output_shape = [-1, 4, 4, 64*2] if self.data_format == 'NHWC' else [-1, 64*2, 4, 4]
                    output       = tf.reshape(output, output_shape)
                    output       = slim.batch_norm(output)
                    output       = self.pixcnn_gated_nonlinearity(output, labels, biases=False)

                with tf.variable_scope('G0'):
                    output = self.upscale(output, [-1, 8, 8, 32*2])
                    output = self.pixcnn_gated_nonlinearity(output, labels)

                    output = self.upscale(output, [-1, 16, 16, 16*2])
                    output = self.pixcnn_gated_nonlinearity(output, labels)

                    output = self.upscale(output, [-1, 32, 32, 8*2])
                    output = self.pixcnn_gated_nonlinearity(output, labels)

                    output = self.upscale(output, [-1, 64, 64, 4*2])
                    output = self.pixcnn_gated_nonlinearity(output, labels)

                    G0     = tf.nn.tanh(self.upscale(output, [-1, 64, 64, 3]))

                with tf.variable_scope('G1'):
                    output = self.join(output, z)
                    output = self.residual_block(output)
                    output = self.residual_block(output)
                    output = self.upscale(output, [-1, 128, 128, 2])
                    G1     = tf.nn.tanh(self.upscale(output, [-1, 128, 128, 3]))

                with tf.variable_scope('G2'):
                    output = self.join(output, z)
                    output = self.residual_block(output)
                    output = self.residual_block(output)
                    output = self.upscale(output, [-1, 256, 256, 1])
                    G2     = tf.nn.tanh(self.upscale(output, [-1, 256, 256, 3]))

                return G0, G1, G2

    def join(self, input_tensor, z):
        with tf.name_scope('join'):
            input_shape = input_tensor.get_shape()

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

            return output

    def residual_block(self, x):
        with tf.name_scope('residual_block'):
            with slim.arg_scope([slim.conv2d], stride=1, padding='same'):
                Fx  = slim.conv2d(x, 64, kernel_size=3)
                Fx  = slim.batch_norm(Fx)
                Fx  = tf.nn.leaky_relu(Fx)
                Fx  = slim.conv2d(Fx, 64, kernel_size=3)
                Fx  = slim.batch_norm(Fx)

                x   = slim.conv2d(x, 64, kernel_size=1)
                x   = slim.batch_norm(x)

                Fxx = Fx + x
                return tf.nn.leaky_relu(Fxx)

    def upscale(self,
                 input_tensor,
                 output_shape,
                 kernel_size=3,
                 stride=1,
                 batch_norm=True):

        filters = output_shape[-1]
        height  = output_shape[1]
        width   = output_shape[2]

        with tf.name_scope('upsample_%d_%d_%d' % (height, width, filters)):
            input_tensor_shape = input_tensor.get_shape().as_list()

            if self.data_format == 'NCHW':
                is_same_height_width = input_tensor_shape[2] == height and input_tensor_shape[3] == width
            else:
                is_same_height_width = input_tensor_shape[1] == height and input_tensor_shape[2] == width

            # only resize if not the same output height/width
            if not is_same_height_width:
                if self.data_format == 'NCHW':
                    input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])

                output  = tf.image.resize_nearest_neighbor(
                    input_tensor,
                    (height, width)
                )

                if self.data_format == 'NCHW':
                    output = tf.transpose(output, [0, 3, 1, 2])
            else:
                output = input_tensor

            output = slim.conv2d(output, filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding='same')

            if batch_norm:
                output = slim.batch_norm(output)

            return output

    def pixcnn_gated_nonlinearity(self, input_tensor, labels, biases=True):
        with tf.name_scope('pixcnn_gated_nonlinearity'):
            condition_tensor = self.generate_condition(input_tensor, labels, biases)

            if self.data_format == 'NHWC':
                even_input     = input_tensor[:,:,:,::2]
                even_condition = condition_tensor[:,:,:,::2]
                odd_input      = input_tensor[:,:,:,1::2]
                odd_condition  = condition_tensor[:,:,:,1::2]
            else:
                even_input     = input_tensor[:,::2]
                even_condition = condition_tensor[:,::2]
                odd_input      = input_tensor[:,1::2]
                odd_condition  = condition_tensor[:,1::2]

            even_tensor    = even_input + even_condition
            odd_tensor     = odd_input + odd_condition

            return tf.sigmoid(even_tensor) * tf.tanh(odd_tensor)

    def generate_condition(self, input_tensor, labels, biases):
        with tf.name_scope('condition'):
            flat_shape = int(np.prod(input_tensor.get_shape()[1:]))
            if biases:
                output = slim.fully_connected(labels, flat_shape, activation_fn=None)
            else:
                output = slim.fully_connected(labels, flat_shape, activation_fn=None, biases_initializer=None)
            output = tf.reshape(output, tf.shape(input_tensor))
            return output
