import tensorflow as tf

def nhwc_to_nchw(net):
    return tf.transpose(net, [0, 3, 1, 2])

def nchw_to_nhwc(net):
    return tf.transpose(net, [0, 2, 3, 1])
