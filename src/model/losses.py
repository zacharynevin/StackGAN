import tensorflow as tf
import tensorflow.contrib.slim as slim

def G_loss(G_logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logits, labels=tf.ones_like(G_logits)))

def D_loss(D_logits, G_logits):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits))
        +tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logits, labels=tf.ones_like(G_logits))
    )

def interpolates(real_batch, fake_batch):
    with tf.variable_scope('interpolates'):
        real_batch = slim.flatten(real_batch)
        fake_batch = slim.flatten(fake_batch)

        alpha = tf.random_uniform([tf.shape(real_batch)[0], 1], minval=0., maxval=1.)

        differences  = fake_batch - real_batch
        return real_batch + (alpha*differences)

def lambda_gradient_penalty(logits, diff):
    with tf.variable_scope('lambda_gradient_penalty'):
        gradients = tf.gradients(logits, [diff])[0]
        slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return 10*gradient_penalty

def wasserstein(real_batch, fake_batch, discrim_func, discrim_scope):
    with tf.name_scope('wasserstein_loss'):
        diff = interpolates(real_batch, fake_batch)
        diff_reshaped = tf.reshape(diff, tf.shape(real_batch))
        interp_logits, _ = discrim_func(diff_reshaped, discrim_scope)

        return lambda_gradient_penalty(interp_logits, diff)

def colour_consistency_regularization(G1, G0):
    with tf.name_scope('cc_regularization'):
        lambda_1 = 1.0
        lambda_2 = 5.0
        alpha    = 50.0

        G1       = slim.flatten(G1)
        G0       = slim.flatten(G0)

        mu_si_j  = tf.reduce_mean(G1, -1)
        mu_si1_j = tf.reduce_mean(G0, -1)

        G1_mu = G1 - tf.expand_dims(mu_si_j, 1)
        G0_mu = G0 - tf.expand_dims(mu_si1_j, 1)

        G0_cov_matrix = tf.matmul(G0_mu, G0_mu, transpose_b=True)
        G0_cov_matrix = G0_cov_matrix / G0.get_shape().as_list()[1]

        G1_cov_matrix = tf.matmul(G1_mu, G1_mu, transpose_b=True)
        G1_cov_matrix = G1_cov_matrix / G1.get_shape().as_list()[1]

        cov_si_j  = tf.reduce_mean(G1_cov_matrix, -1)
        cov_si1_j = tf.reduce_mean(G0_cov_matrix, -1)

        L_ci  = lambda_1 * tf.norm(mu_si_j - mu_si1_j)**2
        L_ci += lambda_2 * tf.norm(cov_si_j - cov_si1_j)**2

        return alpha * tf.reduce_mean(L_ci)
