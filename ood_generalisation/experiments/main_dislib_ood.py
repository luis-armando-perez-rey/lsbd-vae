import tensorflow as tf
import tensorflow_hub as hub
import os
import sys
import time
from pathlib import Path
import pickle
import numpy as np

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

from lsbd_vae.data_utils.data_loader import load_factor_data

from ood_generalisation.modules import presets, utils, evaluation_dislib, data_selection


def reload_encoder_decoder_dislib(result_path, image_shape, dataset, model_name, repetition):
    """
    Reload models which were previously trained using disentanglement_lib https://github.com/google-research/disentanglement_lib
    Args:
        result_path: path where all results are saved
        image_shape: shape of the input images
        dataset: name of dataset e.g. arrow, square, dsprites_lsbd, shapes3d_lsbd, arrow_1_16, arrow_quadrant, ...
        model_name: name of previously trained model e.g. vae, betavae, dip_vae, dip_vae2, factor_vae, tcvae.
        repetition: number of repetition

    Returns:

    """
    model_path = os.path.join(result_path, f"{dataset}_{model_name}_{repetition}", model_name, model_name, "tfhub")
    # Define the encoder
    input_layer = tf.keras.layers.Input(image_shape)
    encoder_layer = hub.KerasLayer(model_path, signature="gaussian_encoder", signature_outputs_as_dict=True)(
        input_layer)
    encoder = tf.keras.models.Model(input_layer, encoder_layer["mean"])

    # Define the decoder
    latent_input_layer = tf.keras.layers.Input(encoder_layer["mean"].shape[-1])  # get the latent variable shape
    decoder_layer = hub.KerasLayer(model_path, signature="decoder", signature_outputs_as_dict=True)(latent_input_layer)
    x_out = tf.keras.activations.sigmoid(decoder_layer["images"])
    decoder = tf.keras.models.Model(latent_input_layer, x_out)
    return encoder, decoder


def reload_elbo_model_dislib(result_path, image_shape, dataset, model_name, repetition):
    model_path = os.path.join(result_path, f"{dataset}_{model_name}_{repetition}", model_name, model_name, "tfhub")
    # Define the encoder
    input_layer = tf.keras.layers.Input(image_shape)
    encoder_layer = hub.KerasLayer(model_path, signature="gaussian_encoder", signature_outputs_as_dict=True)(
        input_layer)
    mean = encoder_layer["mean"]
    logvar = encoder_layer["logvar"]

    # Define the decoder
    decoder_layer = hub.KerasLayer(model_path, signature="decoder", signature_outputs_as_dict=True)(mean)
    x_out = decoder_layer["images"]

    # compute loss components
    flattened_dim = np.prod(image_shape)
    x_in_flat = tf.reshape(input_layer, shape=(-1, flattened_dim))
    x_out_flat = tf.reshape(x_out, shape=(-1, flattened_dim))
    reconstruction_losses = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=x_out_flat, labels=x_in_flat),
        axis=1)
    kl_losses = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - logvar - 1, axis=1)
    elbos = - reconstruction_losses - kl_losses
    return tf.keras.models.Model(input_layer, [reconstruction_losses, kl_losses, elbos])


def evaluate_dislib(result_path, image_shape, dataset, model_name, repetition, neptune_run=None):
    # region =setup data class=
    print("... setting up data class")
    data_parameters, factor_ranges, use_angles_for_selection = presets.DATA_RANGES_ANGLES[dataset]
    dataset_class = load_factor_data(root_path=ROOT_PATH, **data_parameters)
    correct_dsprites_symmetries = True
    # endregion

    # region =split up in training data and ood data=
    print("... setting up training & ood data")
    images, images_train, images_ood, \
        factor_values_as_angles, factor_values_as_angles_train, factor_values_as_angles_ood, \
        factor_values_as_angles_grid = data_selection.split_up_data_ood(dataset_class, data_parameters,
                                                                        correct_dsprites_symmetries,
                                                                        use_angles_for_selection, factor_ranges)
    # endregion

    # region =setup models and evaluate=
    print("... setting up models")
    encoder, decoder = reload_encoder_decoder_dislib(result_path, image_shape, dataset, model_name, repetition)
    loss_model = reload_elbo_model_dislib(result_path, image_shape, dataset, model_name, repetition)
    save_path = Path(result_path, f"{dataset}_{model_name}_{repetition}", model_name, model_name, "evaluations")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    latent_dim = encoder.output.shape[-1]

    # reconstructions of training data
    print("... plotting reconstructions")
    n_samples = 10
    if not utils.file_exists(save_path / "reconstructions_train.png"):
        indices = np.random.choice(len(images_train), size=n_samples, replace=False)
        x_samples = images_train[indices]
        evaluation_dislib.plot_reconstructions(encoder, decoder, x_samples, save_path / "reconstructions_train",
                                               neptune_run)
    # reconstructions of ood data
    if images_ood is not None and not utils.file_exists(save_path / "reconstructions_ood.png"):
        indices = np.random.choice(len(images_ood), size=n_samples, replace=False)
        x_samples = images_ood[indices]
        evaluation_dislib.plot_reconstructions(encoder, decoder, x_samples, save_path / "reconstructions_ood",
                                               neptune_run)

    # 2d embeddings (in 2d grid for 2 factors and 2 latent dimensions)
    print("... plotting latent traversals & embeddings")
    if not utils.file_exists(save_path / "2d_traversals_0_1.png", skip=True):  # only check for first file
        x_dim_f, y_dim_f = presets.DATASET_INTERESTING_LATENT_DIMS[dataset]
        for i in range(latent_dim-1):
            for j in range(i+1, latent_dim):
                evaluation_dislib.plot_2d_latent_traverals(decoder, latent_dim, 10,
                                                           save_path / f"2d_traversals_{i}_{j}", neptune_run,
                                                           x_dim=i, y_dim=j)
                evaluation_dislib.plot_2d_embedding(dataset_class.images, factor_values_as_angles_grid, encoder,
                                                    save_path / f"2d_embedding_{i}_{j}", neptune_run,
                                                    x_dim_f=x_dim_f, y_dim_f=y_dim_f, x_dim_l=i, y_dim_l=j)

    # ood detection plots & auc scores
    # check if density plots file exists, if yes assume all OOD files exist and skip this
    if images_ood is not None and not utils.file_exists(save_path / "ood_detection_dens.pdf"):
        print("... plotting OOD detection plots & computing AUC scores")
        evaluation_dislib.ood_detection(loss_model, images_train, images_ood, save_path / "ood_detection", neptune_run)

    # calculate D_LSBD metric
    print("... computing d_lsbd metric")
    if not utils.file_exists(save_path / "d_lsbd.p"):
        evaluation_dislib.compute_d_lsbd(encoder, dataset_class.images, dataset_class.n_factors, save_path / "d_lsbd",
                                         neptune_run)
    # endregion


def main(kwargs_dislib):
    use_neptune = False
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    if use_neptune:
        import neptune.new as neptune
        from neptune_config import api_key
        neptune_project = "TUe/ood-lsbd"
        neptune_name = "DISLIB"
        print(f"\n=== Logging to Neptune project {neptune_project}, run {timestamp} ==")
        neptune_run_ = neptune.init(project=neptune_project, api_token=api_key.API_KEY, name=neptune_name)
        neptune_run_["parameters"] = kwargs_dislib
        neptune_run_["timestamp"] = timestamp
    else:
        neptune_run_ = None

    print(f"\n=== Experiment timestamp: {timestamp} ===")

    print("\n=== Experiment kwargs_dislib: ===")
    for key, value in kwargs_dislib.items():
        print(key, "=", value)

    # check and print if GPU is used
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    for gpu in physical_devices:
        print(gpu.name)

    print("\n=== Start evaluation ===")
    start_time = time.time()
    evaluate_dislib(neptune_run=neptune_run_, **kwargs_dislib)
    print("\n=== Evaluation done ===")
    end_time = time.time()
    utils.print_and_log_time(start_time, end_time, neptune_run_, "time_elapsed/evaluate")
    print()
    if neptune_run_ is not None:
        neptune_run_.stop()


if __name__ == "__main__":
    BW = (64, 64, 1)
    RGB = (64, 64, 3)
    result_path_ = os.path.join(ROOT_PATH, "results_dislib", "results_models_data_trainingsteps30k_latentdim7")
    repetitions = 1
    dataset_list = [
        # "arrow", "square", "dsprites_lsbd", "shapes3d_lsbd",
        "arrow_1_16", "arrow_quadrant", "arrow_9_16",
        "square_1_16", "square_quadrant", "square_9_16",
        # "dsprites_rte", "dsprites_rtr", "dsprites_extr",
        # "shapes3d_rte", "shapes3d_rtr", "shapes3d_extr",
    ]
    image_shape_list = [
        # RGB, BW, BW, RGB,
        RGB, RGB, RGB,
        BW, BW, BW,
        # BW, BW, BW,
        # RGB, RGB, RGB,
    ]
    model_name_list = ["vae", "betavae", "dip_vae", "dip_vae2", "factor_vae", "tcvae"]

    # OVERWRITE TO RUN NEW DATASETS
    dataset_list = ["arrow_125", "arrow_375", "arrow_625", "arrow_875",
                    "square_125", "square_375", "square_625", "square_875"]
    image_shape_list = [RGB, RGB, RGB, RGB,
                        BW, BW, BW, BW]

    for dataset_, image_shape_ in zip(dataset_list, image_shape_list):
        for model_name_ in model_name_list:
            for repetition_ in range(repetitions):
                kwargs_dislib_ = {
                    "result_path": result_path_,
                    "dataset": dataset_,
                    "image_shape": image_shape_,
                    "model_name": model_name_,
                    "repetition": repetition_
                }
                main(kwargs_dislib_)
                print("Done!")
