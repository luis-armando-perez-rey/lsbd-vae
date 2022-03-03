import os
import tensorflow as tf
import sys
import time
from pathlib import Path
import pickle
import numpy as np

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

# local imports
from lsbd_vae.data_utils.data_loader import load_factor_data
from lsbd_vae.models.lsbd_vae import LSBDVAE
from lsbd_vae.models.architectures import encoder_decoder_dense
from lsbd_vae.models.latentspace import HyperSphericalLatentSpace

from ood_generalisation.modules import presets, utils, evaluation, data_selection


def run_lsbdvae(save_path, data_parameters, factor_ranges, epochs, neptune_run=None):
    # region =setup data_class and latent spaces=
    dataset_class = load_factor_data(root_path=ROOT_PATH, **data_parameters)
    latent_spaces = [HyperSphericalLatentSpace(1), HyperSphericalLatentSpace(1)]
    input_shape = dataset_class.image_shape
    latent_dim = sum([ls.latent_dim for ls in latent_spaces])
    # endregion

    # region =split up in training data and ood data=
    images = dataset_class.flat_images
    factor_values = dataset_class.flat_factor_mesh_as_angles
    indices_ood, indices_train = data_selection.select_factor_combinations(factor_values, factor_ranges)
    images_train = images[indices_train]
    factor_values_train = factor_values[indices_train]
    images_ood = images[indices_ood]
    factor_values_ood = factor_values[indices_ood]
    # endregion

    # region =setup (semi-)supervised train dataset=
    n_labels = len(images_train) // 2  # fully supervised
    x_l, x_l_transformations, x_u = data_selection.setup_circles_dataset_labelled_pairs(images_train,
                                                                                        factor_values_train, n_labels)
    print("X_l shape", x_l.shape, "num transformations", len(x_l_transformations), "x_u len", len(x_u))
    print("transformations shape:", x_l_transformations[0].shape)
    # endregion

    # region =SETUP LSBD-VAE=
    callbacks = []
    if neptune_run is not None:
        neptune_callback = utils.NeptuneMonitor(neptune_run)
        callbacks.append(neptune_callback)
    encoder, decoder = encoder_decoder_dense(latent_dim=latent_dim, input_shape=input_shape)
    lsbdvae = LSBDVAE([encoder], decoder, latent_spaces, 2, input_shape=input_shape)
    lsbdvae.s_lsbd.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    lsbdvae.s_lsbd.fit(x={"images": x_l, "transformations": x_l_transformations}, epochs=epochs, callbacks=callbacks)
    # endregion

    # region =EVALUATIONS=
    # reconstructions of training data
    print("... plotting reconstructions")
    n_samples = 10
    x_l_flat = x_l.reshape((-1, *x_l.shape[2:]))
    indices = np.random.choice(len(x_l_flat), size=n_samples, replace=False)
    x_samples = x_l_flat[indices]
    evaluation.plot_reconstructions(lsbdvae.u_lsbd, x_samples, save_path / "reconstructions_train", neptune_run)
    # reconstructions of ood data
    indices = np.random.choice(len(images_ood), size=n_samples, replace=False)
    x_samples = images_ood[indices]
    evaluation.plot_reconstructions(lsbdvae.u_lsbd, x_samples, save_path / "reconstructions_ood", neptune_run)

    # latent traversals (in 2d grid for 2 factors)
    print("... plotting torus embeddings")
    evaluation.plot_2d_torus_embedding(dataset_class, lsbdvae.u_lsbd, save_path / "2d_embedding", neptune_run)

    # density plots for training vs ood
    print("... plotting latent traversals")
    evaluation.plot_2d_latent_traverals_torus(lsbdvae.u_lsbd, 10, save_path / "2d_traversals_torus", neptune_run)

    # print("... plotting density plots")
    # endregion


def main(kwargs_lsbdvae):
    use_neptune = True
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

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

    print(f"\n=== Experiment timestamp: {timestamp} ===")

    print("\n=== Experiment kwargs_lsbdvae: ===")
    for key, value in kwargs_lsbdvae.items():
        print(key, "=", value)

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    for gpu in physical_devices:
        print(gpu.name)

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
        "data_parameters": presets.SQUARE_PARAMETERS,
        # "data_parameters": presets.ARROW_PARAMETERS,
        "factor_ranges": ((1.5 * np.pi, 2 * np.pi), (1.5 * np.pi, 2 * np.pi)),
        "epochs": 1,
    }

    main(kwargs_lsbdvae_)
    print("Done!")
