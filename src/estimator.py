import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu import TPUEstimatorSpec as EstimatorSpec
from tensorflow.estimator import ModeKeys
from tensorflow.contrib import summary
import numpy as np
import os
from src.config import config
from src.dataset import get_dataset_iterator
from src.model.discriminator import StackGANDiscriminator as Discriminator
from src.model.generator import StackGANGenerator as Generator
import src.model.losses as losses

def model_fn(features, labels, mode, params):
    use_tpu      = params['use_tpu']
    D_lr         = params["D_lr"]
    G_lr         = params["G_lr"]
    num_classes  = params["num_classes"]
    data_format  = params["data_format"]

    loss         = None
    train_op     = None
    predictions  = None
    host_call    = None
    eval_metrics = None

    generator      = Generator(num_classes, data_format)
    discriminator  = Discriminator(num_classes, data_format)

    if mode == ModeKeys.PREDICT:
        G0, G1, G2 = generator(features, labels)

        predictions = {
            'G0': G0,
            'G1': G1,
            'G2': G2
        }
    elif mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        R0       = features['R0']
        R1       = features['R1']
        R2       = features['R2']
        z        = features['z']
        R_labels = labels['R']
        G_labels = labels['G']

        global_step    = tf.train.get_or_create_global_step()
        G_global_step  = tf.Variable(0, dtype=tf.int64, trainable=False, name='G_global_step')
        D0_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D0_global_step')
        D1_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D1_global_step')
        D2_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D2_global_step')

        G0, G1, G2 = generator(z, G_labels)

        D_R0_uncond, D_R0_cond, \
        D_R1_uncond, D_R1_cond, \
        D_R2_uncond, D_R2_cond = discriminator(R0, R1, R2, R_labels)

        D_G0_uncond, D_G0_cond, \
        D_G1_uncond, D_G1_cond, \
        D_G2_uncond, D_G2_cond = discriminator(G0, G1, G2, G_labels)

        op = tf.group(
            tf.assign_add(global_step, 1),
            tf.assign_add(G_global_step, 1),
            tf.assign_add(D0_global_step, 1),
            tf.assign_add(D1_global_step, 1),
            tf.assign_add(D2_global_step, 1)
        ) if mode == ModeKeys.EVAL else tf.no_op()

        with tf.control_dependencies([op]):
            with tf.variable_scope('losses'):
                with tf.variable_scope('G0'):
                    L_G0 = losses.G_loss(D_G0_uncond, D_G0_cond, G_labels)

                with tf.variable_scope('G1'):
                    L_G1  = losses.G_loss(D_G1_uncond, D_G1_cond, G_labels)
                    L_G1 += losses.colour_consistency_regularization(G1, G0)

                with tf.variable_scope('G2'):
                    L_G2  = losses.G_loss(D_G2_uncond, D_G2_cond, G_labels)
                    L_G2 += losses.colour_consistency_regularization(G2, G1)

                with tf.variable_scope('D0'):
                    L_D0 = losses.D_loss(D_R0_uncond, D_R0_cond, R_labels, L_G0)

                with tf.variable_scope('D1'):
                    L_D1 = losses.D_loss(D_R1_uncond, D_R1_cond, R_labels, L_G1)

                with tf.variable_scope('D2'):
                    L_D2 = losses.D_loss(D_R2_uncond, D_R2_cond, R_labels, L_G2)

                with tf.variable_scope('G'):
                    L_G  = L_G0 + L_G1 + L_G2

        if mode == ModeKeys.TRAIN:
            with tf.variable_scope('optimizers'):
                trainable_vars = tf.trainable_variables()
                G_vars  = [var for var in trainable_vars if 'generator' in var.name]
                D0_vars = [var for var in trainable_vars if 'discriminator/D0' in var.name]
                D1_vars = [var for var in trainable_vars if 'discriminator/D1' in var.name]
                D2_vars = [var for var in trainable_vars if 'discriminator/D2' in var.name]

                D0_train = create_train_op(L_D0,
                                           global_step=D0_global_step,
                                           learning_rate=D_lr,
                                           var_list=D0_vars,
                                           use_tpu=use_tpu)

                with tf.control_dependencies([D0_train]):
                    D1_train = create_train_op(L_D1,
                                               global_step=D1_global_step,
                                               learning_rate=D_lr,
                                               var_list=D1_vars,
                                               use_tpu=use_tpu)

                with tf.control_dependencies([D1_train]):
                    D2_train = create_train_op(L_D2,
                                               global_step=D2_global_step,
                                               learning_rate=D_lr,
                                               var_list=D2_vars,
                                               use_tpu=use_tpu)

                with tf.control_dependencies([D2_train]):
                    G_train = create_train_op(L_G,
                                              global_step=G_global_step,
                                              learning_rate=G_lr,
                                              var_list=G_vars,
                                              use_tpu=use_tpu)

                train_op = tf.group(G_train, D2_train, D1_train, D0_train)

                with tf.control_dependencies([train_op]):
                    tf.assign_add(global_step, 1)

        loss = L_G

        host_call = (host_call_fn(mode), [
            G0,
            G1,
            G2,
            R0,
            R1,
            R2,
            tpu_pad(L_D0),
            tpu_pad(L_D1),
            tpu_pad(L_D2),
            tpu_pad(L_G0),
            tpu_pad(L_G1),
            tpu_pad(L_G2),
            tpu_pad(L_G),
            tpu_pad(D0_global_step),
            tpu_pad(D1_global_step),
            tpu_pad(D2_global_step),
            tpu_pad(G_global_step)
        ])

    return EstimatorSpec(mode,
                         predictions=predictions,
                         loss=loss,
                         host_call=host_call,
                         eval_metrics=eval_metrics,
                         train_op=train_op)

def tpu_pad(scalar):
    return tf.reshape(scalar, [1])

def tpu_depad(tensor, dtype=None):
    tensor = tf.reduce_mean(tensor)
    if dtype:
        tensor = tf.cast(tensor, dtype)
    return tensor

def host_call_fn(mode):
    """
    This is a hack for getting multiple losses to appear in Tensorboard.
    It also gives us the ability to write summaries even when using TPUs.
    """
    def summary_fn(G0, G1, G2, R0, R1, R2, L_D0, L_D1, L_D2, L_G0, L_G1, L_G2, L_G,
                     D0_global_step, D1_global_step, D2_global_step, G_global_step):
        with summary.create_file_writer(os.path.join(config.log_dir, mode)).as_default():
            with summary.always_record_summaries():
                max_image_outputs = 10

                D0_global_step = tpu_depad(D0_global_step)
                D1_global_step = tpu_depad(D1_global_step)
                D2_global_step = tpu_depad(D2_global_step)
                G_global_step  = tpu_depad(G_global_step)
                L_D0           = tpu_depad(L_D0)
                L_D1           = tpu_depad(L_D1)
                L_D2           = tpu_depad(L_D2)
                L_G0           = tpu_depad(L_G0)
                L_G1           = tpu_depad(L_G1)
                L_G2           = tpu_depad(L_G2)
                L_G            = tpu_depad(L_G)

                summary.image('R0', R0, max_images=max_image_outputs, step=D0_global_step)
                summary.image('R1', R1, max_images=max_image_outputs, step=D1_global_step)
                summary.image('R2', R2, max_images=max_image_outputs, step=D2_global_step)
                summary.image('G0', G0, max_images=max_image_outputs, step=G_global_step)
                summary.image('G1', G1, max_images=max_image_outputs, step=G_global_step)
                summary.image('G2', G2, max_images=max_image_outputs, step=G_global_step)

                with tf.name_scope('losses'):
                    summary.scalar('D0', L_D0, step=D0_global_step)
                    summary.scalar('D1', L_D1, step=D1_global_step)
                    summary.scalar('D2', L_D2, step=D2_global_step)
                    summary.scalar('G0', L_G0, step=G_global_step)
                    summary.scalar('G1', L_G1, step=G_global_step)
                    summary.scalar('G2', L_G2, step=G_global_step)
                    summary.scalar('G', L_G,   step=G_global_step)

                return summary.all_summary_ops()

    return summary_fn

def predict_input_fn(params, class_label=None):
    sample_size = params['batch_size']
    num_classes = params['num_classes']

    z = tf.random_normal([sample_size, z_dim])

    if class_label is None:
        label = tf.random_uniform(shape=[sample_size], minval=0, maxval=num_classes-1, dtype=tf.int32)
    else:
        label = tf.ones([sample_size]) * class_label

    labels = tf.one_hot(label, num_classes, dtype=tf.float32)

    return z, label

def eval_input_fn(params):
    return get_dataset(params, 'eval')

def train_input_fn(params):
    return get_dataset(params, 'train')

def get_dataset(params, mode):
    batch_size         = params['batch_size']
    buffer_size        = params['buffer_size']
    data_dir           = params['data_dir']
    data_format        = params['data_format']
    num_classes        = params['num_classes']
    z_dim              = params['z_dim']
    seed               = params['data_seed']*2 if mode == 'eval' else params['data_seed']
    num_parallel_calls = params['data_map_parallelism']

    iterator = get_dataset_iterator(data_dir,
                                    batch_size,
                                    num_classes=num_classes,
                                    data_format=data_format,
                                    buffer_size=buffer_size,
                                    shuffle_seed=seed,
                                    num_parallel_calls=num_parallel_calls)

    R0, R1, R2, R_labels = iterator.get_next()

    z        = tf.random_normal([batch_size, z_dim], seed=seed)
    label    = tf.random_uniform(shape=[batch_size], seed=seed, minval=0, maxval=num_classes-1, dtype=tf.int32)
    G_labels = tf.one_hot(label, num_classes, dtype=tf.float32)

    features = {
        'R0': R0,
        'R1': R1,
        'R2': R2,
        'z': z
    }

    labels = {
        'G': G_labels,
        'R': R_labels
    }

    return features, labels

def create_train_op(loss, learning_rate, var_list, global_step, use_tpu=False):
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=10000,
                                                   decay_rate=0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate,
                                       beta1=0.5,
                                       beta2=0.9)

    if use_tpu:
        optimizer = tpu.CrossShardOptimizer(optimizer)

    return optimizer.minimize(loss,
                              var_list=var_list,
                              global_step=global_step,
                              colocate_gradients_with_ops=True)
