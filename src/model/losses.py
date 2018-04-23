import tensorflow as tf
import tensorflow.contrib.slim as slim
import src.model.layers as layers

def G_loss(G_logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logits, labels=tf.ones_like(G_logits)))

def D_loss(D_logits, G_logits):
    output  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=true_labels(D_logits)))
    output += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logits, labels=false_labels(G_logits)))
    return output

def false_labels(labels):
    return tf.random_uniform(tf.shape(labels), .0, .3)

def true_labels(labels):
    return tf.random_uniform(tf.shape(labels), .8, 1.2)

def interpolates(real_batch, fake_batch):
    with tf.name_scope('interpolates'):
        real_batch = slim.flatten(real_batch)
        fake_batch = slim.flatten(fake_batch)
        alpha = tf.random_uniform([tf.shape(real_batch)[0], 1], minval=0., maxval=1.)
        differences  = fake_batch - real_batch
        return real_batch + (alpha*differences)

def lambda_gradient_penalty(logits, diff):
    with tf.name_scope('lambda_gradient_penalty'):
        gradients = tf.gradients(logits, [diff])[0]
        slopes    = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        return 10*gradient_penalty

def wasserstein_loss(real_batch, fake_batch, discrim_func, discrim_scope):
    return 0

    with tf.name_scope('wasserstein_loss'):
        diff = interpolates(real_batch, fake_batch)
        diff_reshaped = tf.reshape(diff, tf.shape(real_batch))
        interp_logits, _ = discrim_func(diff_reshaped, discrim_scope)

        return lambda_gradient_penalty(interp_logits, diff)

def image_mean(img):
    with tf.name_scope('image_mean'):
        img_shape = img.get_shape().as_list()
        channels  = img_shape[1]
        pixels    = img_shape[2] * img_shape[3]

        mu = tf.reduce_mean(img, [2, 3], keepdims=True)
        img_mu = tf.reshape(img - mu, [-1, channels, pixels])
        return mu, img_mu, pixels

def image_covariance(img_mu, pixels):
    with tf.name_scope('image_covariance'):
        cov_matrix = tf.matmul(img_mu, img_mu, transpose_b=True)
        cov_matrix = cov_matrix / pixels

        return cov_matrix

def colour_consistency_regularization(G1, G0, data_format):
    with tf.name_scope('cc_regularization'):
        lambda_1 = 1.0
        lambda_2 = 5.0
        alpha    = 50.0

        if data_format == 'NHWC':
            G0 = layers.nhwc_to_nchw(G0)
            G1 = layers.nhwc_to_nchw(G1)

        mu_si1_j, G0_mu, G0_pixels = image_mean(G0)
        mu_si_j, G1_mu, G1_pixels  = image_mean(G1)

        cov_si1_j = image_covariance(G0_mu, pixels=G0_pixels)
        cov_si_j  = image_covariance(G1_mu, pixels=G1_pixels)

        L_ci  = lambda_1 * tf.losses.mean_squared_error(mu_si_j, mu_si1_j)
        L_ci += lambda_2 * tf.losses.mean_squared_error(cov_si_j, cov_si1_j)

        return alpha * tf.reduce_mean(L_ci)
