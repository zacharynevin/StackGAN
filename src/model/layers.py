import tensorflow as tf
import tensorflow.contrib.slim as slim

def nhwc_to_nchw(net):
    return tf.transpose(net, [0, 3, 1, 2])

def nchw_to_nhwc(net):
    return tf.transpose(net, [0, 2, 3, 1])

def resize(net, dims, data_format):
    height = dims[0]
    width  = dims[1]

    net_shape = net.get_shape().as_list()

    if data_format == 'NCHW':
        net_height = net_shape[2]
        net_width = net_shape[3]
    else:
        net_height = net_shape[1]
        net_width = net_shape[2]

    is_same_height_width = net_height == height and net_width == width

    if is_same_height_width:
        return net

    with tf.name_scope('resize'):
        if data_format == 'NCHW':
            net = nchw_to_nhwc(net)

        net = tf.image.resize_nearest_neighbor(net, (height, width))

        if data_format == 'NCHW':
            net = nhwc_to_nchw(net)

        return net

def conv3x3_block(net, filters, data_format):
    with tf.name_scope('conv3x3_block'):
        net = slim.conv2d(net, filters*2, kernel_size=3, stride=1, padding='same')
        net = slim.batch_norm(net)
        net = glu(net, data_format)
        return net

def glu(net, data_format):
    """
    Gated linear unit
    """
    with tf.name_scope('glu'):
        if data_format == 'NHWC':
            num_channels = net.get_shape().as_list()[-1]
            num_channels = int(num_channels/2)

            net = net[...,:num_channels] * tf.nn.sigmoid(net[...,num_channels:])
        else:
            num_channels = net.get_shape().as_list()[1]
            num_channels = int(num_channels/2)

            net = net[:,:num_channels] * tf.nn.sigmoid(net[:,num_channels:])

        return net
