import os

import tensorflow as tf
import sys

sys.path.append(os.getcwd())
from lsbd_vae.data.data_loader import load_factor_data
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

tf.debugging.experimental.enable_dump_debug_info(
    dump_root='./',
    tensor_debug_mode='FULL_HEALTH',
    circular_buffer_size=883010,
    op_regex="(?!^Const$)"
)

SQUARE_PARAMETERS = {
    "data": "pixel",
    "height": 64,
    "width": 64,
    "step_size_vert": 1,
    "step_size_hor": 1,
    "square_size": 16
}


def test_train_ulsbd_vae():
    dataset_class = load_factor_data(**SQUARE_PARAMETERS)
    latent_spaces = [HyperSphericalLatentSpace(1), HyperSphericalLatentSpace(1)]
    input_shape = (SQUARE_PARAMETERS["height"], SQUARE_PARAMETERS["width"], 1)
    latent_dim = 4
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)

    ulsbd = UnsupervisedLSBDVAE([encoder], decoder, latent_spaces, input_shape=input_shape)
    ulsbd.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    ulsbd.fit(x={"images": dataset_class.images}, epochs=EPOCHS)
    return 0
