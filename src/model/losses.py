import tensorflow as tf
import tensorflow.contrib.slim as slim

def G_loss(D_G_uncond, D_G_cond, G_labels):
    L_G  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_uncond, labels=tf.ones_like(D_G_uncond)))
    L_G += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_cond, labels=G_labels))
    return 0.5*L_G

def D_loss(D_R_uncond, D_R_cond, R_labels, L_G):
    L_D  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_R_uncond, labels=tf.ones_like(D_R_uncond)))
    L_D += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_R_cond, labels=R_labels))
    L_D += -L_G
    return 0.5*L_D

def kl_loss(mu, log_sigma):
    with tf.name_scope('kl_divergence'):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        return tf.reduce_mean(loss)

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
