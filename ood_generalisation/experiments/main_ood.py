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
from lsbd_vae.models.lsbd_vae import SupervisedLSBDVAE
from lsbd_vae.models import architectures, reconstruction_losses
from lsbd_vae.models.latentspace import HyperSphericalLatentSpace

from ood_generalisation.modules import presets, utils, evaluation, data_selection


def get_architectures(architecture: str, **kwargs):
    if architecture == "dense":
        dense_units_lst = (512, 256)
        return architectures.encoder_decoder_dense(dense_units_lst=dense_units_lst, **kwargs)
    elif architecture == "conv":
        filters_lst = (128, 64, 32)
        dense_units_lst = (64,)
        return architectures.encoder_decoder_vgglike_2d(
            filters_lst=filters_lst, dense_units_lst=dense_units_lst, **kwargs)
    elif architecture == "dislib":
        return architectures.encoder_decoder_dislib_2d(**kwargs)
    else:
        raise ValueError(f"{architecture} architecture not defined")


def get_reconstruction_loss(reconstruction_loss):
    if reconstruction_loss == "gaussian":
        return reconstruction_losses.gaussian_loss
    elif reconstruction_loss == "bernoulli":
        return reconstruction_losses.bernoulli_loss
    else:
        raise ValueError(f"{reconstruction_loss} loss not defined")


def run_lsbdvae(save_path: Path, data_parameters: dict, factor_ranges: tuple, epochs: int, batch_size: int,
                supervision_batch_size: int, architecture: str, reconstruction_loss: str,
                log_t_limit: tuple = (-10, -6), neptune_run=None, correct_dsprites_symmetries=False,
                use_angles_for_selection=True, early_stopping=False):
    # region =setup data_class and latent spaces=
    dataset_class = load_factor_data(root_path=ROOT_PATH, **data_parameters)
    latent_spaces = []
    for _ in range(dataset_class.n_factors):
        latent_spaces.append(HyperSphericalLatentSpace(1, log_t_limit=log_t_limit))
    input_shape = dataset_class.image_shape
    latent_dim = sum([ls.latent_dim for ls in latent_spaces])
    # endregion

    # region =split up in training data and ood data=
    images = dataset_class.flat_images

    # regular factor values are needed to select factor combinations given factor_ranges,
    # factor values as angles are needed for LSBD-VAE training
    if correct_dsprites_symmetries and data_parameters["data"] == "dsprites":
        factor_values_grid = dataset_class.factor_mesh
        factor_values_as_angles_grid = dataset_class.factor_mesh_as_angles
        # factor0 is shape: 0=square, 1=ellips, 2=heart. factor2 is orientation (axis5 contains factor values)
        # project angles for square onto 90 degrees, then rescale back to 360 degrees
        factor_values_grid[0, :, :, :, :, 2] = np.mod(factor_values_grid[0, :, :, :, :, 2], 0.5 * np.pi) * 4
        factor_values_as_angles_grid[0, :, :, :, :, 2] = \
            np.mod(factor_values_as_angles_grid[0, :, :, :, :, 2], 0.5 * np.pi) * 4
        # project angles for ellips onto 180 degrees, then rescale back to 360 degrees
        factor_values_grid[1, :, :, :, :, 2] = np.mod(factor_values_grid[1, :, :, :, :, 2], np.pi) * 2
        factor_values_as_angles_grid[1, :, :, :, :, 2] = \
            np.mod(factor_values_as_angles_grid[1, :, :, :, :, 2], np.pi) * 2
        # flatten
        factor_values = np.reshape(factor_values_grid, (-1, 5))
        factor_values_as_angles = np.reshape(factor_values_as_angles_grid, (-1, 5))
    else:
        if data_parameters["data"] == "dsprites":
            print("correct_dsprites_symmetries is set to True, but the data isn't dsprites, ignoring this parameter")
        # factor_values_grid = dataset_class.factor_mesh
        factor_values_as_angles_grid = dataset_class.factor_mesh_as_angles
        factor_values = dataset_class.flat_factor_mesh
        factor_values_as_angles = dataset_class.flat_factor_mesh_as_angles

    if factor_ranges is None:
        images_train = images
        factor_values_as_angles_train = factor_values_as_angles
        images_ood = None
        # factor_values_as_angles_ood = None
    else:
        if use_angles_for_selection:
            indices_ood, indices_train = \
                data_selection.select_factor_combinations(factor_values_as_angles, factor_ranges)
        else:
            indices_ood, indices_train = data_selection.select_factor_combinations(factor_values, factor_ranges)
        images_train = images[indices_train]
        factor_values_as_angles_train = factor_values_as_angles[indices_train]
        images_ood = images[indices_ood]
        # factor_values_as_angles_ood = factor_values_as_angles[indices_ood]
    # endregion

    # region =setup (semi-)supervised train dataset=
    n_labels = len(images_train) // supervision_batch_size  # fully supervised
    x_l, x_l_transformations, x_u = \
        data_selection.setup_circles_dataset_labelled_batches(images_train, factor_values_as_angles_train, n_labels,
                                                              supervision_batch_size)
    print("X_l shape", x_l.shape, "num transformations", len(x_l_transformations), "x_u len", len(x_u))
    print("transformations shape:", x_l_transformations[0].shape)
    # endregion

    # region =SETUP/TRAIN/SAVE LSBD-VAE=
    callbacks = []
    if neptune_run is not None:
        neptune_callback = utils.NeptuneMonitor(neptune_run)
        callbacks.append(neptune_callback)
    if early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss_s", min_delta=1, patience=3,
                                                                   mode="min", restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping_callback)
    encoder, decoder = get_architectures(architecture, latent_dim=latent_dim, input_shape=input_shape)
    print("\n=== Encoder Architecture: ===")
    encoder.summary()
    print("\n=== Decoder Architecture: ===")
    decoder.summary()
    print()
    lsbdvae = SupervisedLSBDVAE([encoder], decoder, latent_spaces, supervision_batch_size, input_shape=input_shape,
                                reconstruction_loss=get_reconstruction_loss(reconstruction_loss))
    lsbdvae.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    lsbdvae.fit(x={"images": x_l, "transformations": x_l_transformations}, epochs=epochs, batch_size=batch_size,
                callbacks=callbacks)

    lsbdvae.save_weights(save_path / "model_weights")
    # endregion

    # region =EVALUATIONS=
    # reconstructions of training data
    print("... plotting reconstructions")
    n_samples = 10
    indices = np.random.choice(len(images_train), size=n_samples, replace=False)
    x_samples = images_train[indices]
    evaluation.plot_reconstructions(lsbdvae, x_samples, save_path / "reconstructions_train", neptune_run)
    # reconstructions of ood data
    if images_ood is not None:
        indices = np.random.choice(len(images_ood), size=n_samples, replace=False)
        x_samples = images_ood[indices]
        evaluation.plot_reconstructions(lsbdvae, x_samples, save_path / "reconstructions_ood", neptune_run)

    # torus embeddings & latent traversals (in 2d grid for 2 factors)
    print("... plotting torus embeddings & latent traversals")
    for i in range(dataset_class.n_factors-1):
        for j in range(i+1, dataset_class.n_factors):
            evaluation.plot_2d_latent_traverals_torus(lsbdvae, 10, save_path / f"2d_traversals_torus_{i}_{j}",
                                                      neptune_run, x_dim=i, y_dim=j)
            evaluation.plot_2d_torus_embedding(dataset_class.images, factor_values_as_angles_grid, lsbdvae,
                                               save_path / f"2d_embedding_{i}_{j}", neptune_run, x_dim=i, y_dim=j)

    # circle embeddings, one for each latent space
    print("... plotting circle embeddings")
    evaluation.plot_circle_embeddings(images, factor_values_as_angles, lsbdvae, save_path / "circle_embeddings",
                                      neptune_run)

    # density plots for training vs ood
    if images_ood is not None:
        print("... plotting OOD detection plots")
        evaluation.ood_detection(lsbdvae, images_train, images_ood, save_path / "ood_detection", neptune_run)
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
    if neptune_run_ is not None:
        neptune_run_.stop()


if __name__ == "__main__":
    kwargs_lsbdvae_ = {
        # "data_parameters": presets.SQUARE_PARAMETERS, "use_angles_for_selection": True,
        "data_parameters": presets.ARROW_PARAMETERS, "use_angles_for_selection": True,
        "factor_ranges": presets.FACTOR_RANGES_2D_9_16,
        # "data_parameters": {"data": "dsprites"}, "use_angles_for_selection": False,
        # "factor_ranges": presets.FACTOR_RANGES_DSPRITES_RTE,
        # "data_parameters": {"data": "shapes3d"}, "use_angles_for_selection": False,
        # "factor_ranges": presets.FACTOR_RANGES_SHAPES3D_EXTR,
        "epochs": 50,
        "batch_size": 8,
        "supervision_batch_size": 32,
        "architecture": "dislib",  # "dense", "conv", "dislib"
        "reconstruction_loss": "bernoulli",  # "gaussian", "bernoulli"
        "log_t_limit": (-10, -6),
        "correct_dsprites_symmetries": True,
        "early_stopping": True,
    }

    main(kwargs_lsbdvae_)
    print("Done!")
