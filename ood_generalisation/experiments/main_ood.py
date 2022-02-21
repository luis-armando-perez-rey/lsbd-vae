import os
import tensorflow as tf
import sys
import time
from pathlib import Path
import pickle

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# local imports
from lsbd_vae.data_utils.data_loader import load_factor_data
from lsbd_vae.models.lsbd_vae import LSBDVAE
from lsbd_vae.models.architectures import encoder_decoder_dense
from lsbd_vae.models.latentspace import HyperSphericalLatentSpace

from ood_generalisation.modules import presets, utils


def run_lsbdvae(save_path, data_parameters, epochs, neptune_run=None):
    callbacks = []
    if neptune_run is not None:
        neptune_callback = utils.NeptuneMonitor(neptune_run)
        callbacks.append(neptune_callback)

    dataset_class = load_factor_data(**data_parameters)
    latent_spaces = [HyperSphericalLatentSpace(1), HyperSphericalLatentSpace(1)]
    input_shape = dataset_class.image_shape
    latent_dim = 4  # TODO: check if this needs to be hardcoded? seems dependent on latent spaces
    n_labels = dataset_class.n_data_points // 2
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)
    x_l, x_l_transformations, x_u = dataset_class.setup_circles_dataset_labelled_pairs(n_labels=n_labels)
    print("X_l shape", x_l.shape, "num transformations", len(x_l_transformations), "x_u len", len(x_u))
    print("transformations shape:", x_l_transformations[0].shape)
    lsbdvae = LSBDVAE([encoder], decoder, latent_spaces, 2, input_shape=input_shape)
    lsbdvae.s_lsbd.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    lsbdvae.s_lsbd.fit(x={"images": x_l, "transformations": x_l_transformations}, epochs=epochs, callbacks=callbacks)


def main(kwargs_lsbdvae):
    use_neptune = True

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    print(f"\n=== Experiment timestamp: {timestamp} ===")

    print("\n=== Experiment kwargs_lsbdvae: ===")
    for key, value in kwargs_lsbdvae.items():
        print(key, "=", value)

    if use_neptune:
        import neptune.new as neptune
        from neptune_config import api_key
        neptune_project = "TUe/ood-lsbd"
        neptune_name = "LSBD-VAE"
        print(f"\n=== Logging to Neptune project {neptune_project}, run {timestamp} ==")
        neptune_run_ = neptune.init(project=neptune_project, api_token=api_key.API_KEY, name=neptune_name)
        neptune_run_["parameters"] = kwargs_lsbdvae
        neptune_run_["timestamp"] = timestamp
    else:
        neptune_run_ = None

    save_path = Path("results", timestamp)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    settings_path = save_path / "settings.p"
    with open(settings_path, "wb") as f:
        print("... saving kwargs_lsbdvae to", settings_path)
        pickle.dump(kwargs_lsbdvae, f)

    print("\n=== Start training ===")
    start_time = time.time()
    run_lsbdvae(save_path=save_path, neptune_run=neptune_run_, **kwargs_lsbdvae)
    print("\n=== Training done ===")
    end_time = time.time()
    utils.print_and_log_time(start_time, end_time, neptune_run_, "time_elapsed/train")
    print()


if __name__ == "__main__":
    kwargs_lsbdvae_ = {
        "epochs": 2,
        "data_parameters": presets.SQUARE_PARAMETERS,
    }

    main(kwargs_lsbdvae_)
    print("Done!")
