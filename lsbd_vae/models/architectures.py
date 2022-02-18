from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape, \
    BatchNormalization, Conv2DTranspose
from tensorflow.keras.applications import ResNet50V2
from typing import Tuple
from tensorflow.keras.models import Model
import numpy as np


def get_encoder_decoder(architecture: str, image_shape: Tuple[int, int, int], latent_dim: int):
    if architecture == "dense":
        architecture_parameters = {"input_shape": image_shape,
                                   "dense_units_lst": (512, 512, 256, 100),
                                   "latent_dim": latent_dim}
        architecture_function = encoder_decoder_dense
    elif architecture == "resnet50v2_dense":
        architecture_parameters = {"encoder_params": {"include_top": False,
                                                      "weights": "imagenet",
                                                      "pooling": "avg",
                                                      },
                                   "decoder_params":
                                       {"input_shape": image_shape,
                                        "dense_units_lst": (512, 512, 256),
                                        "latent_dim":latent_dim}
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
    h = Dense(np.prod(input_shape), activation="sigmoid")(h)
    x_reconstr = Reshape(input_shape)(h)
    decoder = Model(dec_in, x_reconstr)

    return encoder, decoder



