from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import tensorflow as tf


def gaussian_loss(dec_std=1 / (2 ** 0.5), n_data_dims=3):
    dec_std = tf.cast(dec_std, tf.float32)  # cannot take K.log of an int

    def loss(x_in, x_out):
        return tf.reduce_sum(tf.square(x_in - x_out) / (2 * dec_std ** 2) +
                             tf.math.log(dec_std) + 0.5 * tf.math.log(2 * np.pi),
                             axis=(*range(-n_data_dims, 0),))

    return loss


def bernoulli_loss(n_data_dims=3):
    def loss(x_in, x_out):
        # BCE computes a mean over the final dimension, but we want a sum over all data_dims instead,
        # so we fix this by multiplying with the size of the final dimension. Furthermore, BCE reduces the final dim,
        # so we only need to reduce_sum over the last n_data_dims-1 dimensions.
        image_depth = x_in.shape[-1]
        return tf.reduce_sum(binary_crossentropy(x_in, x_out) * image_depth,
                             axis=(*range(-n_data_dims+1, 0),))
    return loss
