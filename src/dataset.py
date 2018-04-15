import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def get_dataset_iterator(data_dir,
                         batch_size,
                         num_classes,
                         data_format,
                         buffer_size,
                         shuffle_seed,
                         num_parallel_calls):
    """Construct a TF dataset from a remote source"""
    def transform(tfrecord_proto):
        return transform_tfrecord(tfrecord_proto,
                                  num_classes=num_classes,
                                  data_format=data_format)

    tf_dataset  = tf.data.TFRecordDataset(data_dir)
    tf_dataset  = tf_dataset.map(transform, num_parallel_calls=num_parallel_calls)
    tf_dataset  = tf_dataset.shuffle(seed=shuffle_seed, buffer_size=buffer_size)
    tf_dataset  = tf_dataset.repeat()
    tf_dataset  = tf_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    tf_iterator = tf_dataset.make_one_shot_iterator()
    return tf_iterator

def decode_image(image_buff, height, width):
    """
    Take a raw image byte string and decode to an image

    Params:
        image_buff (str): Image byte string
        height (int): The original image height
        width (int): The original image width

    Return:
        Tensor[NCHW | NHCW]: A tensor of shape representing the RGB image.
    """
    image = tf.decode_raw(image_buff, out_type=tf.uint8)
    image = tf.reshape(image, tf.stack([height, width, 3], axis=0))
    image = tf.reverse(image, [-1]) # BGR to RGB
    return image

def resize_and_convert_image(image, dims, data_format):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_nearest_neighbor(image, dims)
    if data_format == 'NCHW':
        image = tf.transpose(image, [0, 3, 1, 2])
    image = tf.squeeze(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def decode_class(label, num_classes):
    return tf.one_hot(label, num_classes, dtype=tf.float32)

def transform_tfrecord(tf_protobuf, num_classes, data_format):
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
        "image": tf.FixedLenFeature((), tf.string),
        "height": tf.FixedLenFeature([], tf.int64),
        "width": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(tf_protobuf, features)

    decoded_image = decode_image(parsed_features["image"],
                                 height=parsed_features["height"],
                                 width=parsed_features["width"])

    image_64  = resize_and_convert_image(decoded_image, [64, 64], data_format=data_format)
    image_128 = resize_and_convert_image(decoded_image, [128, 128], data_format=data_format)
    image_256 = resize_and_convert_image(decoded_image, [256, 256], data_format=data_format)

    decoded_class = decode_class(parsed_features["label"], num_classes)

    return image_64, image_128, image_256, decoded_class
