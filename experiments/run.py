import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pickle

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "lsbd_vae"))
from lsbd_vae.utils.model_utils import get_autoencoder_model, get_ls_list
from lsbd_vae.models.architectures import get_encoder_decoder
from lsbd_vae.metrics.dlsbd_metric import dlsbd, create_combinations_omega_values_range
from lsbd_vae.utils import plotting
from lsbd_vae.data_utils.data_loader import load_factor_data
from experiments.configs.paper_dataset_parameters import return_data_parameters
from lsbd_vae.data_utils.factor_dataset import FactorImageDataset

TORUS_DATASETS = {"arrow", "pixel4", "modelnet_colors", "pixel8", "pixel16"}

tf.config.run_functions_eagerly(True)

# region ARGUMENTS SCRIPT
parser = argparse.ArgumentParser(
    description="Training of LSBDVAE model either using paths or with semi-supervised pairs")
parser.add_argument("--gpu", nargs="?", dest="gpu_num", type=str, default="0", help="gpu to be used")
parser.add_argument("--dataset", nargs="?", dest="dataset", type=str, default="", help="Dataset to be used")
parser.add_argument("--architecture", nargs="?", dest="architecture", type=str, default="dis_lib",
                    help="Architecture of model")
parser.add_argument("--tag", nargs="?", dest="tag", type=str, default="test",
                    help="Experiment tag used to save")
parser.add_argument("--npaths", nargs="?", dest="npaths", type=int, default=50,
                    help="Number of paths")
parser.add_argument("--lpaths", nargs="?", dest="lpaths", type=int, default=100,
                    help="Length of paths")
parser.add_argument("--spaths", nargs="?", dest="spaths", type=int, default=3,
                    help="Step size of paths")
parser.add_argument("--epochs", nargs="?", dest="epochs", type=int, default=100,
                    help="Training epochs")
parser.add_argument("--batch_size", nargs="?", dest="batch_size", type=int, default=100,
                    help="Batch size")
parser.add_argument("--model_type", nargs="?", dest="model_type", type=str, default="LSBDVAE",
                    help="Autoencoder model type")
parser.add_argument("--logt", nargs="?", dest="logt", type=int, default=-5,
                    help="Clipping value for the upper bound of the log-scale")
parser.add_argument("--distweight", nargs="?", dest="distweight", type=float, default=1,
                    help="Value for the weighting factor of the equivariance loss in the LSBDVAE")
parser.add_argument('--trainpath', dest="trainpath", action='store_true')

parser.add_argument('--no-trainpath', dest="trainpath", action='store_false')
parser.set_defaults(trainpath=True)
parser.add_argument("--npairs", dest="npairs", type=int, default=0,
                    help="When trainpath is False, then train semi-supervised using npairs of transform-labelled data")

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
# endregion
base_result_path = "./experiments/results"
saving_folder = os.path.join(base_result_path, args.tag, args.dataset, args.model_type)
experiment_metrics_save_path = os.path.join(saving_folder, "metrics")
plot_save_path = os.path.join(saving_folder, "plots")
os.makedirs(experiment_metrics_save_path, exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

# region LATENT SPACES PARAMETERS
if args.dataset in TORUS_DATASETS:
    latent_dim = 4
    latent_parameters = {
        "latent_types": ["s", "s"],
        "latent_dims": [1, 1],
        "kl_weights": None,
        "kwargs_list": [{"log_t_limit": (-10, args.logt), "dist_weight": args.distweight},
                        {"log_t_limit": (-10, args.logt), "dist_weight": args.distweight}]
    }
else:
    latent_dim = 7
    latent_parameters = {
        "latent_types": ["s", "e"],
        "latent_dims": [1, 5],
        "kl_weights": None,
        "kwargs_list": [{"log_t_limit": (-10, args.logt), "dist_weight": args.distweight},
                        {"dist_weight": args.distweight}]
    }
# endregion

# region DATA LOADING
data_parameters = return_data_parameters(args.dataset)
data_class: FactorImageDataset = load_factor_data(**data_parameters)
if args.dataset in TORUS_DATASETS:
    if args.trainpath:
        x_l, x_l_transformations = data_class.random_paths_torus(args.npaths, args.lpaths, args.spaths)
        x_u = np.array([])
        n_transforms = args.lpaths
    else:
        n_transforms = args.npairs
        x_l, x_l_transformations, x_u = data_class.setup_circles_dataset_labelled_pairs(args.npairs)
else:
    if args.trainpath:
        x_l, x_l_transformations = data_class.random_paths_cylinder(args.npaths, args.lpaths, args.spaths)
        x_u = np.array([])
        n_transforms = args.lpaths
    else:
        x_l, x_l_transformations, x_u = data_class.setup_cylinder_dataset_labelled_pairs(args.npairs)
        n_transforms = args.npairs

print("Unlabeled shape", x_u.shape)
print("Labeled shape", x_l.shape, "Transformation shape 1", x_l_transformations[0].shape)
# endregion

# region MODEL INITIALIZATION
model_parameters = {
    "model_type": args.model_type,
    "n_transforms": n_transforms,
    "input_shape": data_class.image_shape
}
architecture_params = {
    "latent_dim": latent_dim,
    "architecture": args.architecture,
    "image_shape": data_class.image_shape
}
ls_list = get_ls_list(**latent_parameters)
encoder_backbone, decoder_backbone = get_encoder_decoder(**architecture_params)
model_class = get_autoencoder_model(latent_spaces=ls_list, encoder_backbones=[encoder_backbone],
                                    decoder_backbone=decoder_backbone,
                                    **model_parameters)
model_class.compile(optimizer="adam")
# endregion

# region TRAIN MODEL
callbacks = [tf.keras.callbacks.TerminateOnNaN()]
model_class.fit_semi_supervised(x_l, x_l_transformations, x_u, epochs=args.epochs, callback_list=callbacks,
                                batch_size=args.batch_size)
# endregion

# region SAVE MODEL

# save parameters
save_path = os.path.join(saving_folder, args.tag)
os.makedirs(save_path, exist_ok=True)
reload_parameters = {"model_parameters": model_parameters, "latent_parameters": latent_parameters,
                     "architecture_parameters": architecture_params}
params_path = os.path.join(save_path, "parameters.pkl")
with open(params_path, 'wb') as f:
    pickle.dump(reload_parameters, f)

# model_class.predict(x_l)
weight_path = os.path.join(save_path, "weights")
print(f"Saving model weights in {save_path}")
model_class.save_weights(weight_path)
print("Testing load of model weights...")
model_class.load_weights(weight_path)
print("Model weights saved and loaded")
# endregion

# region ENCODE IMAGES
encoded_list = model_class.encode_images(data_class.flat_images)
scale_list = model_class.encode_images_scale(data_class.flat_images)
latent_reps = np.concatenate(encoded_list, axis=-1)
print("Latent representation shape", np.expand_dims(latent_reps, axis=1).shape)
reconstructions = model_class.generate_images(encoded_list)
latent_reps = latent_reps.reshape((*data_class.factors_shape, latent_reps.shape[-1]))
# endregion

# region PLOTTING
# ----------- PLOTTING ----------
print("START PLOTTING")
# plot some reconstructions
print("...plotting reconstructions")
sample_size = 20  # sample size
print("Flat image shape", data_class.flat_images.shape)
print("Reconstructions shape", reconstructions.shape)
x_and_rec = np.stack([data_class.flat_images, reconstructions], axis=1)
indices = np.random.choice(len(x_and_rec), size=sample_size, replace=False)
x_sample = np.moveaxis(x_and_rec[indices], 0, 1)
fig = plotting.plot_subset(x_sample)
fig.savefig(os.path.join(plot_save_path, f"reconstructions.png"))
plt.show()
plt.close()

print("Plotting embeddings PCA")
for num_encoding, encoding in enumerate(encoded_list):
    print("Encodings shape", encoding.shape)
    fig, ax = plotting.plot_objects_pca(data_class.flat_images, encoding, zoom=0.3)
    ax.grid()
    fig.savefig(os.path.join(plot_save_path, f"{num_encoding}_pca.png"))
    plt.show()
    plt.close(fig)

if args.dataset in TORUS_DATASETS:
    print("Plot unwrapped torus")
    unwrapped_angles = np.stack([np.arctan2(encoding[:, 1], encoding[:, 0]) for encoding in encoded_list], axis=-1)
    fig, ax = plotting.plot_objects_pca(data_class.flat_images, unwrapped_angles, zoom=0.3)
    ax.grid()
    ax.set_xlabel("Angle1")
    ax.set_xlabel("Angle2")
    fig.savefig(os.path.join(plot_save_path, "torus_angles.png"))
    plt.show()
    plt.close(fig)

# endregion

# region LSBD METRIC
print("Start calculation of LSBD metric")
if args.dataset in TORUS_DATASETS:
    omega_values = create_combinations_omega_values_range(start_value=-10, end_value=10)
    lsbd_score, _ = dlsbd(latent_reps, omega_values, be_verbose=True, factor_manifold="torus")
else:
    omega_values = create_combinations_omega_values_range(start_value=-10, end_value=10, n_transforms=1)
    lsbd_score, _ = dlsbd(latent_reps, omega_values, be_verbose=True, factor_manifold="cylinder")

# Saving folder
np.save(os.path.join(experiment_metrics_save_path, "lsbd.npy"), lsbd_score)

print("METRIC LSBD Score", lsbd_score)
# endregion
