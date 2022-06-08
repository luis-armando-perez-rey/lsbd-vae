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
    print("... setting up data class")
    dataset_class = load_factor_data(root_path=ROOT_PATH, **data_parameters)
    latent_spaces = []
    for _ in range(dataset_class.n_factors):
        latent_spaces.append(HyperSphericalLatentSpace(1, log_t_limit=log_t_limit))
    input_shape = dataset_class.image_shape
    latent_dim = sum([ls.latent_dim for ls in latent_spaces])
    # endregion

    # region =split up in training data and ood data=
    print("... setting up training & ood data")
    images, images_train, images_ood, \
        factor_values_as_angles, factor_values_as_angles_train, factor_values_as_angles_ood, \
        factor_values_as_angles_grid = data_selection.split_up_data_ood(dataset_class, data_parameters,
                                                                        correct_dsprites_symmetries,
                                                                        use_angles_for_selection, factor_ranges)
    # endregion

    # region =SETUP LSBD-VAE=
    print("... setting up model")
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
    # endregion

    if (save_path / "model_weights.index").is_file() is False:
        # region =setup (semi-)supervised train dataset=
        print("... training new model")
        n_labels = len(images_train) // supervision_batch_size  # fully supervised
        x_l, x_l_transformations, x_u = \
            data_selection.setup_circles_dataset_labelled_batches(images_train, factor_values_as_angles_train, n_labels,
                                                                  supervision_batch_size)
        print("X_l shape", x_l.shape, "num transformations", len(x_l_transformations), "x_u len", len(x_u))
        print("transformations shape:", x_l_transformations[0].shape)
        # endregion

        # region =TRAIN+SAVE LSBD-VAE=
        lsbdvae.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
        lsbdvae.fit(x={"images": x_l, "transformations": x_l_transformations}, epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks)

        lsbdvae.save_weights(save_path / "model_weights")
        # endregion

    else:  # load weights from previously trained model
        print("... loading weights of previously trained model")
        lsbdvae.load_weights(save_path / "model_weights")

    # region =EVALUATIONS=
    # reconstructions of training data
    print("... plotting reconstructions")
    n_samples = 10
    if not utils.file_exists(save_path / "reconstructions_train.png"):
        indices = np.random.choice(len(images_train), size=n_samples, replace=False)
        x_samples = images_train[indices]
        evaluation.plot_reconstructions(lsbdvae, x_samples, save_path / "reconstructions_train", neptune_run)
    # reconstructions of ood data
    if images_ood is not None and not utils.file_exists(save_path / "reconstructions_ood.png"):
        indices = np.random.choice(len(images_ood), size=n_samples, replace=False)
        x_samples = images_ood[indices]
        evaluation.plot_reconstructions(lsbdvae, x_samples, save_path / "reconstructions_ood", neptune_run)

    # torus embeddings & latent traversals (in 2d grid for 2 factors)
    print("... plotting torus embeddings & latent traversals")
    if not utils.file_exists(save_path / "2d_traversals_torus_0_1.png", overwrite=True):  # only check for first file
        for i in range(dataset_class.n_factors-1):
            for j in range(i+1, dataset_class.n_factors):
                evaluation.plot_2d_latent_traverals_torus(lsbdvae, 10, save_path / f"2d_traversals_torus_{i}_{j}",
                                                          neptune_run, x_dim=i, y_dim=j)
                evaluation.plot_2d_torus_embedding(dataset_class.images, factor_values_as_angles_grid, lsbdvae,
                                                   save_path / f"2d_embedding_{i}_{j}", neptune_run, x_dim=i, y_dim=j)
                evaluation.plot_2d_torus_embedding(dataset_class.images, factor_values_as_angles_grid, lsbdvae,
                                                   save_path / f"2d_embedding_{i}_{j}_ood", neptune_run,
                                                   x_dim=i, y_dim=j, factor_ranges=factor_ranges)

    # circle embeddings, one for each latent space
    print("... plotting circle embeddings")
    if not utils.file_exists(save_path / "circle_embeddings_0.pdf"):  # only check for first file
        evaluation.plot_circle_embeddings(images, factor_values_as_angles, lsbdvae, save_path / "circle_embeddings",
                                          neptune_run)

    # density plots for training vs ood
    # check if density plots file exists, if yes assume all OOD files exist and skip this
    if images_ood is not None and not utils.file_exists(save_path / "ood_detection_dens.pdf", overwrite=False):
        print("... plotting OOD detection plots & computing AUC scores")
        evaluation.ood_detection(lsbdvae, images_train, images_ood, save_path / "ood_detection", neptune_run)

    # calculate D_LSBD metric
    print("... computing d_lsbd metric")
    if not utils.file_exists(save_path / "d_lsbd.p", skip=True):
        evaluation.compute_d_lsbd(lsbdvae, dataset_class.images, dataset_class.n_factors, save_path / "d_lsbd",
                                  neptune_run)
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

    # check and print if GPU is used
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    for gpu in physical_devices:
        print(gpu.name)

    save_path = Path("results", timestamp)
    save_path.mkdir(parents=True, exist_ok=True)
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
