from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, \
    BatchNormalization, Conv2DTranspose
from tensorflow.keras.applications import ResNet50V2
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


def get_encoder_decoder(architecture: str, image_shape: Tuple[int, int, int], latent_dim: int):
    if architecture == "dense":
        architecture_parameters = {"input_shape": image_shape,
                                   "dense_units_lst": (512, 512, 256, 100),
                                   "latent_dim": latent_dim}
        architecture_function = encoder_decoder_dense

    elif architecture == "dislib":
        architecture_parameters = {"height": image_shape[0],
                                   "width": image_shape[1],
                                   "depth": image_shape[2],
                                   "latent_dim": latent_dim}
        architecture_function = encoder_decoder_dislib_2d
    elif architecture == "resnet50v2_dense":
        architecture_parameters = {"encoder_params": {"include_top": False,
                                                      "weights": "imagenet",
                                                      "pooling": "avg",
                                                      },
                                   "decoder_params":
                                       {"input_shape": image_shape,
                                        "dense_units_lst": (512, 512, 256),
                                        "latent_dim": latent_dim}
                                   }

        def architecture_function(encoder_params, decoder_params):
            encoder_preload = ResNet50V2(**encoder_params)
            encoder_preload.trainable = False
            input_layer = Input(image_shape)
            x = encoder_preload(input_layer)
            x = Dense(1000, activation="relu")(x)
            encoder = Model(input_layer, x)
            _, decoder = encoder_decoder_dense(**decoder_params)
            return encoder, decoder
    elif architecture == "resnet50v2_dense_trainable":
        architecture_parameters = {"encoder_params": {"include_top": False,
                                                      "weights": "imagenet",
                                                      "pooling": "avg",
                                                      },
                                   "decoder_params":
                                       {"input_shape": image_shape,
                                        "dense_units_lst": (512, 512, 256),
                                        "latent_dim": latent_dim}
                                   }

        def architecture_function(encoder_params, decoder_params):
            encoder_preload = ResNet50V2(**encoder_params)
            input_layer = Input(image_shape)
            x = encoder_preload(input_layer)
            encoder = Model(input_layer, x)
            print("Encoder shape", tf.shape(encoder.input))
            encoder.trainable = True
            [_, decoder] = encoder_decoder_dense(**decoder_params)
            return encoder, decoder

    elif architecture == "simple_cnn":
        architecture_parameters = {"input_shape": image_shape,
                                   "features": 96,
                                   "latent_dim": latent_dim}
        architecture_function = get_encoder_decoder_simple_cnn

    else:
        raise ValueError(f"{architecture} not defined")

    encoder_backbone, decoder_backbone = architecture_function(**architecture_parameters)
    return encoder_backbone, decoder_backbone


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
    h = Dense(np.prod(input_shape), activation=None)(h)
    x_reconstr = Reshape(input_shape)(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder


def encoder_decoder_vgglike_2d(latent_dim: int, input_shape: Tuple = (64, 64, 1), activation: str = "relu",
                               filters_lst: Tuple = (128, 64, 32), dense_units_lst: Tuple = (64,),
                               kernel_size: int = 3, pool_size: int = 2):
    """
    Simple but general convolutional architecture. Uses fixed kernel_size & pool_size. Customisable nr of conv2d layers,
    optional dense layers. Uses MaxPooling and UpSampling.
    """
    n_conv_layers = len(filters_lst)
    height, width, depth = input_shape
    # calculate sizes after convolutions (before dense layers)
    conv_height = height // (pool_size ** n_conv_layers)
    conv_width = width // (pool_size ** n_conv_layers)
    conv_depth = filters_lst[-1]
    assert height == conv_height * (pool_size ** n_conv_layers), "height must be a multiple of pool_size^n_conv_layers"
    assert width == conv_width * (pool_size ** n_conv_layers), "width must be a multiple of pool_size^n_conv_layers"

    # ENCODER
    x = Input(shape=input_shape)
    # convolutional layers
    h = x
    for i in range(n_conv_layers):
        h = Conv2D(filters=filters_lst[i], kernel_size=kernel_size, strides=(1, 1), padding="same",
                   activation=activation)(h)
        h = MaxPooling2D(pool_size=pool_size, padding="same")(h)
    # dense layers
    h = Flatten()(h)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(latent_dim,))
    # dense layers
    h = dec_in
    for units in reversed(dense_units_lst):
        h = Dense(units, activation=activation)(h)
    h = Dense(conv_height * conv_width * conv_depth, activation=activation)(h)
    h = Reshape((conv_height, conv_width, conv_depth))(h)
    # convolutional layers
    for i in reversed(range(1, n_conv_layers)):
        h = UpSampling2D(size=pool_size)(h)
        h = Conv2D(filters=filters_lst[i - 1], kernel_size=kernel_size, strides=(1, 1), padding="same",
                   activation=activation)(h)
    h = UpSampling2D(size=pool_size)(h)
    x_reconstr = Conv2D(filters=depth, kernel_size=kernel_size, strides=(1, 1), padding="same",
                        activation="sigmoid")(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder


def encoder_decoder_dislib_2d(latent_dim: int, input_shape: Tuple = (64, 64, 1), activation: str = "relu",
                              filters_lst: Tuple = (32, 32, 64, 64), dense_units_lst: Tuple = (256,),
                              kernel_size: int = 4, strides: int = 2):
    """
    Simple but general convolutional architecture. Uses fixed kernel_size & pool_size. Customisable nr of conv2d layers,
    optional dense layers. Uses strided (transposed) convolutions.
    If used with default settings, this corresponds to the convolutional architecture from disentanglement_lib.
    """
    n_conv_layers = len(filters_lst)
    height, width, depth = input_shape
    # calculate sizes after convolutions (before dense layers)
    conv_height = height // (strides ** n_conv_layers)
    conv_width = width // (strides ** n_conv_layers)
    conv_depth = filters_lst[-1]
    assert height == conv_height * (strides ** n_conv_layers), "height must be a multiple of pool_size^n_conv_layers"
    assert width == conv_width * (strides ** n_conv_layers), "width must be a multiple of pool_size^n_conv_layers"

    # ENCODER
    x = Input(shape=input_shape)
    # convolutional layers
    h = x
    for i in range(n_conv_layers):
        h = Conv2D(filters=filters_lst[i], kernel_size=kernel_size, strides=strides, padding="same",
                   activation=activation)(h)
    # dense layers
    h = Flatten()(h)
    for units in dense_units_lst:
        h = Dense(units, activation=activation)(h)
    encoder = Model(x, h)

    # DECODER
    dec_in = Input(shape=(latent_dim,))
    # dense layers
    h = dec_in
    for units in reversed(dense_units_lst):
        h = Dense(units, activation=activation)(h)
    h = Dense(conv_height * conv_width * conv_depth, activation=activation)(h)
    h = Reshape((conv_height, conv_width, conv_depth))(h)
    # convolutional layers
    for i in reversed(range(1, n_conv_layers)):
        h = Conv2DTranspose(filters=filters_lst[i - 1], kernel_size=kernel_size, strides=strides, padding="same",
                            activation=activation)(h)
    x_reconstr = Conv2DTranspose(filters=depth, kernel_size=kernel_size, strides=strides, padding="same",
                                 activation="sigmoid")(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder


def get_encoder_decoder_simple_cnn(input_shape, latent_dim, features=96):
    im_input = Input(shape=input_shape)
    C1 = Conv2D(features, 5, padding='valid', activation='relu')(im_input)
    P1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(C1)

    C2 = Conv2D(int(features / 2), 5, padding='valid', activation='relu')(P1)
    P2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(C2)

    C3 = Conv2D(int(features / 4), 5, padding='valid', activation='relu')(P2)
    P3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(C3)

    flat = Flatten()(P3)
    encoder = Model(im_input, flat)

    # Decoder

    spat_dims = P3.shape.as_list()[1:]

    latent_input = Input(shape=(latent_dim,))
    H1 = Dense(np.prod(spat_dims), activation='relu')(latent_input)
    square = Reshape(spat_dims)(H1)

    TP1 = UpSampling2D(size=(2, 2))(square)
    CT1 = Conv2D(int(features / 2), 7, padding='valid', activation='relu')(TP1)

    TP2 = UpSampling2D(size=(2, 2))(CT1)
    CT2 = Conv2D(features, 7, padding='valid', activation='relu')(TP2)

    TP3 = UpSampling2D(size=(2, 2))(CT2)
    CT3 = Conv2D(features, 7, padding='valid', activation='relu')(TP3)

    TP4 = UpSampling2D(size=(2, 2))(CT3)
    CT4 = Conv2D(input_shape[-1], 7, padding='valid', activation='tanh', name='recons')(TP4)

    decoder = Model(latent_input, CT4)

    return encoder, decoder
