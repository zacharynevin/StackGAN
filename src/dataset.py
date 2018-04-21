import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def get_dataset_iterator(data_dir,
                         batch_size,
                         data_format,
                         buffer_size,
                         shuffle_seed,
                         num_parallel_calls):
    """Construct a TF dataset from a remote source"""
    def transform(tfrecord_proto):
        return transform_tfrecord(tfrecord_proto,
                                  data_format=data_format)

    tf_dataset  = tf.data.TFRecordDataset(data_dir)
    tf_dataset  = tf_dataset.map(transform, num_parallel_calls=num_parallel_calls)
    tf_dataset  = tf_dataset.shuffle(seed=shuffle_seed, buffer_size=buffer_size)
    tf_dataset  = tf_dataset.repeat()
    tf_dataset  = tf_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    tf_iterator = tf_dataset.make_one_shot_iterator()
    return tf_iterator

def decode_image(img, dim, data_format):
    """
    Take a raw image byte string and decode to an image

    Params:
        img (str): Image byte string
        dim (int): The width and height of the image.
        data_format (str): The data format for the image

    Return:
        Tensor[HCW]: A tensor of shape representing the RGB image.
    """
    img = tf.decode_raw(img, out_type=tf.uint8)
    img = tf.reshape(img, tf.stack([dim, dim, 3], axis=0))
    img = tf.reverse(img, [-1]) # BGR to RGB
    img = transform_image(img, data_format)
    return img

def transform_image(img, data_format):
    img = tf.image.convert_image_dtype(img, tf.float32)
    if data_format == 'NCHW':
        img = tf.transpose(img, [3, 1, 2])

    return img

def decode_class(label, num_classes):
    return tf.one_hot(label, num_classes, dtype=tf.float32)

def transform_tfrecord(tf_protobuf, data_format):
    """
    Decode the tfrecord protobuf into the image.

    Params:
        tf_protobuf (proto): A protobuf representing the data record.

    Returns:
        Tensor[64, 64, 3]
        Tensor[128, 128, 3]
        Tensor[256, 256, 3]
    """

    features = {
        "image_64": tf.FixedLenFeature((), tf.string),
        "image_128": tf.FixedLenFeature((), tf.string),
        "image_256": tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(tf_protobuf, features)

    image_64  = decode_image(parsed_features["image_64"], 64, data_format=data_format)
    image_128 = decode_image(parsed_features["image_128"], 128, data_format=data_format)
    image_256 = decode_image(parsed_features["image_256"], 256, data_format=data_format)

    return image_64, image_128, image_256
