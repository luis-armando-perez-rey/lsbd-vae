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
        data_dim = tf.reduce_prod(tf.cast(x_in.shape[-n_data_dims:], tf.float32))
        x_in_flat = tf.reshape(x_in, (*x_in.shape[:-n_data_dims], data_dim))
        x_out_flat = tf.reshape(x_out, (*x_out.shape[:-n_data_dims], data_dim))
        return data_dim * binary_crossentropy(x_in_flat, x_out_flat)
