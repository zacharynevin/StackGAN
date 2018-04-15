import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib import summary
import numpy as np
import os
import src.config as config
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        G0, G1, G2 = generator(features, labels)

        predictions = {
            'G0': G0,
            'G1': G1,
            'G2': G2
        }
    elif mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        R0       = features['R0']
        R1       = features['R1']
        R2       = features['R2']
        z        = features['z']
        R_labels = labels['R']
        G_labels = labels['G']

        G0, G1, G2 = generator(z, G_labels)

        D_R0_uncond, D_R0_cond, \
        D_R1_uncond, D_R1_cond, \
        D_R2_uncond, D_R2_cond = discriminator(R0, R1, R2, R_labels)

        D_G0_uncond, D_G0_cond, \
        D_G1_uncond, D_G1_cond, \
        D_G2_uncond, D_G2_cond = discriminator(G0, G1, G2, G_labels)

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

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('optimizers'):
                trainable_vars = tf.trainable_variables()
                G_vars  = [var for var in trainable_vars if 'generator' in var.name]
                D0_vars = [var for var in trainable_vars if 'discriminator/D0' in var.name]
                D1_vars = [var for var in trainable_vars if 'discriminator/D1' in var.name]
                D2_vars = [var for var in trainable_vars if 'discriminator/D2' in var.name]

                D0_train = create_train_op(L_D0,
                                           learning_rate=D_lr,
                                           var_list=D0_vars,
                                           use_tpu=use_tpu)

                with tf.control_dependencies([D0_train]):
                    D1_train = create_train_op(L_D1,
                                               learning_rate=D_lr,
                                               var_list=D1_vars,
                                               use_tpu=use_tpu)

                with tf.control_dependencies([D1_train]):
                    D2_train = create_train_op(L_D2,
                                               learning_rate=D_lr,
                                               var_list=D2_vars,
                                               use_tpu=use_tpu)

                with tf.control_dependencies([D2_train]):
                    G_train = create_train_op(L_G,
                                              learning_rate=G_lr,
                                              var_list=G_vars,
                                              use_tpu=use_tpu)

            train_op = tf.group(G_train, D2_train, D1_train, D0_train)

        predictions = {
            'G0': G0,
            'G1': G1,
            'G2': G2
        }

        loss = L_D0 + L_D1 + L_D2 + L_G

        host_call = (host_call_fn, [G0, G1, G2, R0, R1, R2, L_G0, L_G1, L_G2, L_D0, L_D1, L_D2, L_G])

        eval_metrics = (metric_fn, [L_D0, L_D1, L_D2, L_G])

    return tf.contrib.tpu.TPUEstimatorSpec(mode,
                                           predictions=predictions,
                                           loss=loss,
                                           host_call=host_call,
                                           eval_metrics=eval_metrics,
                                           train_op=train_op)

def host_call_fn(G0, G1, G2, R0, R1, R2, L_G0, L_G1, L_G2, L_D0, L_D1, L_D2, L_G):
    with summary.create_file_writer(config.log_dir).as_default():
        with summary.always_record_summaries():
            max_image_outputs = 10

            summary.image('R0', R0, max_images=max_image_outputs)
            summary.image('R1', R1, max_images=max_image_outputs)
            summary.image('R2', R2, max_images=max_image_outputs)
            summary.image('G0', G0, max_images=max_image_outputs)
            summary.image('G1', G1, max_images=max_image_outputs)
            summary.image('G2', G2, max_images=max_image_outputs)

            with tf.name_scope('loss'):
                summary.scalar('D0', L_D0)
                summary.scalar('D1', L_D1)
                summary.scalar('D2', L_D2)
                summary.scalar('G0', L_G0)
                summary.scalar('G1', L_G1)
                summary.scalar('G2', L_G2)

            return summary.all_summary_ops()

def metric_fn(G_loss, D0_loss, D1_loss, D2_loss):
    return {
        'loss/G_avg_loss': tf.metrics.mean(G_loss),
        'loss/D0_avg_loss': tf.metrics.mean(D0_loss),
        'loss/D1_avg_loss': tf.metrics.mean(D1_loss),
        'loss/D2_avg_loss': tf.metrics.mean(D2_loss)
    }

def predict_input_fn(params):
    sample_size = params['batch_size']
    num_classes = params['num_classes']

    z        = tf.random_normal([sample_size, z_dim])
    label    = tf.random_uniform(shape=[sample_size], minval=0, maxval=num_classes-1, dtype=tf.int32)
    labels   = tf.one_hot(label, num_classes, dtype=tf.float32)

    return (z, label)

def eval_input_fn(params):
    return get_dataset(params, 'eval')

def train_input_fn(params):
    return get_dataset(params, 'train')

def get_dataset(params, mode):
    batch_size  = params['batch_size']
    buffer_size = params['buffer_size']
    data_dir    = params['data_dir']
    data_format = params['data_format']
    num_classes = params['num_classes']
    z_dim       = params['z_dim']

    iterator = get_dataset_iterator(data_dir,
                                    batch_size,
                                    num_classes=num_classes,
                                    data_format=data_format,
                                    buffer_size=buffer_size)

    R0, R1, R2, R_labels = iterator.get_next()

    z        = tf.random_normal([batch_size, z_dim])
    label    = tf.random_uniform(shape=[batch_size], minval=0, maxval=num_classes-1, dtype=tf.int32)
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

def create_train_op(loss, learning_rate, var_list, use_tpu=False):
    global_step = tf.train.get_or_create_global_step()

    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=100000,
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
