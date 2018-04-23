import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu import TPUEstimatorSpec as EstimatorSpec
from tensorflow.contrib import summary
import numpy as np
import os
from src.config import config
from src.dataset import get_dataset_iterator
from src.model.discriminator import StackGANDiscriminator as Discriminator
from src.model.generator import StackGANGenerator as Generator
import src.model.losses as losses

ModeKeys = tf.estimator.ModeKeys

def model_fn(features, labels, mode, params):
    use_tpu      = params['use_tpu']
    D_lr         = params["D_lr"]
    G_lr         = params["G_lr"]
    data_format  = params["data_format"]

    loss         = None
    train_op     = None
    predictions  = None
    host_call    = None
    eval_metrics = None

    generator      = Generator(data_format)
    discriminator  = Discriminator(data_format)

    if mode == ModeKeys.PREDICT:
        G0, G1, G2 = generator(features)

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

        global_step    = tf.train.get_or_create_global_step()

        G_global_step  = tf.Variable(0, dtype=tf.int64, trainable=False, name='G_{}_global_step'.format(mode))
        D0_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D0_{}_global_step'.format(mode))
        D1_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D1_{}_global_step'.format(mode))
        D2_global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='D2_{}_global_step'.format(mode))

        G0, G1, G2, G_scope = generator(z)

        D_R0, D0_scope = discriminator.D0(R0)
        D_R1, D1_scope = discriminator.D1(R1)
        D_R2, D2_scope = discriminator.D2(R2)

        D_G0, _        = discriminator.D0(G0)
        D_G1, _        = discriminator.D1(G1)
        D_G2, _        = discriminator.D2(G2)

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
                    L_G0 = losses.G_loss(D_G0)

                with tf.variable_scope('G1'):
                    L_G1  = losses.G_loss(D_G1)
                    L_G1 += losses.colour_consistency_regularization(G1, G0, data_format=data_format)

                with tf.variable_scope('G2'):
                    L_G2  = losses.G_loss(D_G2)
                    L_G2 += losses.colour_consistency_regularization(G2, G1, data_format=data_format)

                with tf.variable_scope('D0'):
                    L_D0   = losses.D_loss(D_R0, D_G0)
                    L_D0_W = losses.wasserstein_loss(R0, G0, discriminator.D0, D0_scope)
                    L_D0  += L_D0_W

                with tf.variable_scope('D1'):
                    L_D1   = losses.D_loss(D_R1, D_G1)
                    L_D1_W = losses.wasserstein_loss(R1, G1, discriminator.D1, D1_scope)
                    L_D1  += L_D1_W

                with tf.variable_scope('D2'):
                    L_D2   = losses.D_loss(D_R2, D_G2)
                    L_D2_W = losses.wasserstein_loss(R2, G2, discriminator.D2, D2_scope)
                    L_D2  += L_D2_W

                with tf.variable_scope('G'):
                    L_G  = L_G0 + L_G1 + L_G2

        if mode == ModeKeys.TRAIN:
            with tf.variable_scope('optimizers'):
                with tf.control_dependencies([tf.assign_add(global_step, 1)]):
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

                    D1_train = create_train_op(L_D1,
                                               global_step=D1_global_step,
                                               learning_rate=D_lr,
                                               var_list=D1_vars,
                                               use_tpu=use_tpu)

                    D2_train = create_train_op(L_D2,
                                               global_step=D2_global_step,
                                               learning_rate=D_lr,
                                               var_list=D2_vars,
                                               use_tpu=use_tpu)

                    with tf.control_dependencies([D2_train, D1_train, D0_train]):
                        G_train = create_train_op(L_G,
                                                  global_step=G_global_step,
                                                  learning_rate=G_lr,
                                                  var_list=G_vars,
                                                  use_tpu=use_tpu)

                    train_op = tf.group(G_train, D2_train, D1_train, D0_train)

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
            tpu_pad(L_D0_W),
            tpu_pad(L_D1_W),
            tpu_pad(L_D2_W),
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
    It also gives us the ability to write summaries when using TPUs, which are normally incompatible with tf.summary.
    """
    def summary_fn(G0, G1, G2, R0, R1, R2, L_D0, L_D1, L_D2, L_D0_W, L_D1_W, L_D2_W, L_G0, L_G1, L_G2, L_G,
                     D0_global_step, D1_global_step, D2_global_step, G_global_step):
        with summary.create_file_writer(config.log_dir).as_default():
            with summary.always_record_summaries():
                max_image_outputs = 10

                D0_global_step = tpu_depad(D0_global_step)
                D1_global_step = tpu_depad(D1_global_step)
                D2_global_step = tpu_depad(D2_global_step)
                G_global_step  = tpu_depad(G_global_step)
                L_D0           = tpu_depad(L_D0)
                L_D1           = tpu_depad(L_D1)
                L_D2           = tpu_depad(L_D2)
                L_D0_W         = tpu_depad(L_D0_W)
                L_D1_W         = tpu_depad(L_D1_W)
                L_D2_W         = tpu_depad(L_D2_W)
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
                    summary.scalar('D0_W', L_D0_W, step=D0_global_step)
                    summary.scalar('D1_W', L_D1_W, step=D1_global_step)
                    summary.scalar('D2_W', L_D2_W, step=D2_global_step)

                    summary.scalar('G0', L_G0, step=G_global_step)
                    summary.scalar('G1', L_G1, step=G_global_step)
                    summary.scalar('G2', L_G2, step=G_global_step)
                    summary.scalar('G', L_G,   step=G_global_step)

                return summary.all_summary_ops()

    return summary_fn

def predict_input_fn(params, class_label=None):
    sample_size = params['batch_size']
    z = tf.random_normal([sample_size, z_dim])

    return z, None

def eval_input_fn(params):
    return get_dataset(params, 'eval')

def train_input_fn(params):
    return get_dataset(params, 'train')

def get_dataset(params, mode):
    batch_size         = params['batch_size']
    buffer_size        = params['buffer_size']
    data_dir           = params['data_dir']
    data_format        = params['data_format']
    z_dim              = params['z_dim']
    seed               = params['data_seed']*2 if mode == 'eval' else params['data_seed']
    num_parallel_calls = params['data_map_parallelism']

    iterator = get_dataset_iterator(data_dir,
                                    batch_size,
                                    data_format=data_format,
                                    buffer_size=buffer_size,
                                    shuffle_seed=seed,
                                    num_parallel_calls=num_parallel_calls)

    R0, R1, R2 = iterator.get_next()

    z = tf.random_normal([batch_size, z_dim])

    features = {
        'R0': R0,
        'R1': R1,
        'R2': R2,
        'z': z
    }

    return features, None

def create_train_op(loss, learning_rate, var_list, global_step, use_tpu=False):
    exp_learning_rate = tf.train.exponential_decay(learning_rate,
                                                   global_step,
                                                   decay_steps=10000,
                                                   decay_rate=0.96)

    optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate,
                                       beta1=0.5,
                                       beta2=0.999)

    if use_tpu:
        optimizer = tpu.CrossShardOptimizer(optimizer)

    return optimizer.minimize(loss,
                              var_list=var_list,
                              global_step=global_step,
                              colocate_gradients_with_ops=True)
