from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, \
    BatchNormalization, Conv2DTranspose
from typing import Tuple
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


def encoder_decoder_dense(latent_dim: int, input_shape: Tuple = (28, 28, 1), activation: str = "relu",
                          dense_units_lst: Tuple = (512, 256)) -> Tuple[Model, Model]:
    # ENCODER
    x = Input(shape=input_shape)
    h = Flatten()(x)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
        h = BatchNormalization()(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(latent_dim,))
    h = Dense(dense_units_lst[-1], activation)(dec_in)
    for units in reversed(dense_units_lst[:-1]):
        h = Dense(units, activation=activation)(h)
        h = BatchNormalization()(h)
    h = Dense(np.prod(input_shape), activation="sigmoid")(h)
    x_reconstr = Reshape(input_shape)(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder
