import os

import tensorflow as tf
import sys
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
print(os.path.dirname(os.path.dirname(os.getcwd())))
from lsbd_vae.models.lsbd_vae import UnsupervisedLSBDVAE, SupervisedLSBDVAE, LSBDVAE
from lsbd_vae.models.architectures import encoder_decoder_dense
from lsbd_vae.models.latentspace import HyperSphericalLatentSpace, GaussianLatentSpace

NUM_VIEWS = 12
HEIGHT = 64
WIDTH = 64
CHANNELS = 3
TOTAL_DATA = 10
EPOCHS = 10


# tf.compat.v1.disable_eager_execution()

# tf.debugging.experimental.enable_dump_debug_info(
#     dump_root='./',
#     tensor_debug_mode='FULL_HEALTH',
#     circular_buffer_size=883010,
#     op_regex="(?!^Const$)"
# )


def test_train_ulsbd_vae():
    latent_spaces = [HyperSphericalLatentSpace(1), GaussianLatentSpace(3)]
    input_shape = (HEIGHT, WIDTH, CHANNELS)
    latent_dim = 5
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)
    fake_data = tf.random.normal((TOTAL_DATA, 1, *input_shape))
    fake_data = tf.clip_by_value(fake_data, 0.0, 1.0)
    ulsbd = UnsupervisedLSBDVAE([encoder], decoder, latent_spaces, input_shape=input_shape)
    ulsbd.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    ulsbd.fit(x={"images": fake_data}, epochs=EPOCHS)
    return 0


def test_train_slsbd_vae():
    latent_spaces = [HyperSphericalLatentSpace(1), GaussianLatentSpace(3)]
    input_shape = (HEIGHT, WIDTH, CHANNELS)
    latent_dim = 5
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)
    fake_data = tf.random.normal((TOTAL_DATA, NUM_VIEWS, *input_shape))
    fake_data = tf.clip_by_value(fake_data, 0.0, 1.0)
    fake_transformations = (2 * np.pi * tf.random.uniform((TOTAL_DATA, NUM_VIEWS, 1)),
                            tf.ones((TOTAL_DATA, NUM_VIEWS, 1)))

    ulsbd = SupervisedLSBDVAE([encoder], decoder, latent_spaces, NUM_VIEWS, input_shape=input_shape)
    ulsbd.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    ulsbd.fit(x={"images": fake_data, "transformations": fake_transformations}, epochs=EPOCHS)
    return 0


def test_train_lsbd_vae():
    latent_spaces = [HyperSphericalLatentSpace(1), GaussianLatentSpace(3)]
    input_shape = (HEIGHT, WIDTH, CHANNELS)
    latent_dim = 5
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)
    fake_data_l = tf.random.normal((TOTAL_DATA, NUM_VIEWS, *input_shape))
    fake_data_l = tf.clip_by_value(fake_data_l, 0.0, 1.0)
    fake_data_u = tf.random.normal((TOTAL_DATA, 1, *input_shape))
    fake_data_u = tf.clip_by_value(fake_data_u, 0.0, 1.0)
    fake_transformations = [tf.random.uniform((TOTAL_DATA, NUM_VIEWS, 1)), tf.ones((TOTAL_DATA, NUM_VIEWS, 1))]
    lsbd = LSBDVAE([encoder], decoder, latent_spaces, NUM_VIEWS, input_shape=input_shape)
    lsbd.u_lsbd.compile(optimizer=tf.keras.optimizers.Adam())
    lsbd.s_lsbd.compile(optimizer=tf.keras.optimizers.Adam())
    lsbd.fit_semi_supervised(fake_data_l, fake_transformations, fake_data_u, epochs=EPOCHS)
    return 0


if __name__ == "__main__":
    # test_train_ulsbd_vae()
    test_train_slsbd_vae()
    # test_train_lsbd_vae()
    print("Everything passed")
