import numpy as np
import pickle
import json

from ood_generalisation.modules import plotting


def plot_reconstructions(encoder, decoder, x, filepath, neptune_run=None):
    """
    Args:
        encoder: tf.keras encoder model with one output (the mean)
        decoder: tf.keras decoder model
        x: input images array, shape (n_samples, *input_shape)
        filepath: Path object with filepath (excluding extension) where to save image.
        neptune_run: Neptune.ai run
    """
    filepath = filepath.parent / (filepath.name + ".png")
    x_enc = encoder.predict(x)
    x_recon = decoder.predict(x_enc)
    x_array = np.stack([x, x_recon], axis=0)
    plotting.plot_images_grid(x_array, filepath=filepath, neptune_run=neptune_run)


## TODO: this is for LSBD-VAE, adapt (1D traversals?) or remove
def plot_circle_embeddings(images, factor_values_as_angles, lsbd, filepath, neptune_run=None, n_samples=500):
    """
    F = number of factors (and number of latent spaces)

    Args:
        images: shape (n_images, h, w, d)
        factor_values_as_angles: shape (n_images, F), given as angles
        lsbd: BaseLSBDVAE instance or subclass
        filepath:
        neptune_run:
        n_samples: how many points to plot
    """
    sample_indices = np.random.choice(len(images), size=n_samples, replace=False)
    images_sample = images[sample_indices]
    factor_values_sample = factor_values_as_angles[sample_indices]
    encodings_list = lsbd.encode_images(images_sample)
    for factor in range(factor_values_as_angles.shape[-1]):
        colors = factor_values_sample[:, factor]
        filepath_factor = filepath.parent / (filepath.name + f"_{factor}.pdf")
        plotting.plot_circle_embedding(encodings_list[factor], colors, filepath_factor, neptune_run)


def plot_2d_embedding(images_grid, factor_values_as_angles_grid, encoder, filepath, neptune_run=None,
                      x_dim_f=0, y_dim_f=1, x_dim_l=0, y_dim_l=1):
    """
    F = number of factors (and number of latent spaces)
    This makes most sense for datasets with 2 factors (and ideally 2 latent variables)

    Args:
        images_grid: shape (n1, ..., nF, h, w, d)
        factor_values_as_angles_grid:  shape (n1, ..., nF, F), given as angles
        encoder: tf.keras encoder model with one output (the mean)
        filepath:
        neptune_run:
        x_dim_f: value in [0, F), distinct from y_dim_f
        y_dim_f: value in [0, F), distinct from x_dim_f
        x_dim_l: value in [0, latent_dim), distinct from y_dim_l
        y_dim_l: value in [0, latent_dim), distinct from x_dim_l
    """
    assert x_dim_f != y_dim_f, "first and second factor dimensions should not be the same"
    assert x_dim_l != y_dim_l, "first and second latent dimensions should not be the same"

    for factor, factor_size in enumerate(factor_values_as_angles_grid.shape[:-1]):
        if factor != x_dim_f and factor != y_dim_f:
            random_factor_index = np.random.randint(factor_size)
            # take single element for this factor index, [] brackets ensure that dimension is kept (with size 1)
            images_grid = np.take(images_grid, [random_factor_index], axis=factor)  # size of dim factor is changed to 1
            factor_values_as_angles_grid = np.take(factor_values_as_angles_grid, [random_factor_index], axis=factor)
    flat_images = np.reshape(images_grid, (-1, *images_grid.shape[-3:]))
    flat_factor_mesh = np.reshape(factor_values_as_angles_grid, (-1, factor_values_as_angles_grid.shape[-1]))

    encodings = encoder.predict(flat_images)  # shape (n_datapoints, latent_dim)
    # encoded list is a list of length n_latent_spaces, each item is an array of shape (n, ls_latent_dim)

    v_angle = flat_factor_mesh[:, x_dim_f]
    h_angle = flat_factor_mesh[:, y_dim_f]
    colors = plotting.yiq_embedding(v_angle, h_angle)

    encoded_horizontal_angle = encodings[:, x_dim_l]
    encoded_vertical_angle = encodings[:, y_dim_l]
    plotting.plot_torus_angles(encoded_horizontal_angle, encoded_vertical_angle, colors,
                               filepath=filepath, neptune_run=neptune_run)


def plot_2d_latent_traverals(decoder, latent_dim, n_gridpoints, filepath, neptune_run=None,
                             x_dim=0, y_dim=1, minval=-3, maxval=3):
    assert x_dim != y_dim, "first and second dimensions should not be the same"
    # make linear spaces from minval to maxval (inclusive)
    grid_x = np.linspace(minval, maxval, num=n_gridpoints, endpoint=True)
    grid_y = np.linspace(minval, maxval, num=n_gridpoints, endpoint=True)
    # make list of 2 mesh grids, each has shape (n_gridpoints, n_gridpoints), values varying over axis0 and axis1 resp.
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
    # make mesh grid of shape (n_gridpoints, n_gridpoints, latent_dim) with fixed random values for non-grid latent dims
    grid = np.empty((n_gridpoints, n_gridpoints, latent_dim))
    for i in range(latent_dim):
        if i == x_dim:
            grid[:, :, i] = mesh_x
        elif i == y_dim:
            grid[:, :, i] = mesh_y
        else:
            grid[:, :, i] = np.random.normal()
    grid_flat = np.reshape(grid, (n_gridpoints * n_gridpoints, latent_dim))

    # decode
    x_generated = decoder.predict(grid_flat)  # shape (n_gridpoints*n_gridpoints, *input_dim)
    x_generated = np.reshape(x_generated, (n_gridpoints, n_gridpoints, *x_generated.shape[1:]))
    # plot output
    plotting.plot_images_grid(x_generated, filepath=filepath, neptune_run=neptune_run)


def ood_detection(loss_model, x_normal, x_ood, filepath, neptune_run=None):
    reconstruction_losses_normal, kl_losses_normal, elbos_normal = loss_model.predict(x_normal)
    reconstruction_losses_ood, kl_losses_ood, elbos_ood = loss_model.predict(x_ood)

    filepath_hist = filepath.parent / (filepath.name + "_hist.pdf")
    filepath_dens = filepath.parent / (filepath.name + "_dens.pdf")
    filepath_roc = filepath.parent / (filepath.name + "_roc.pdf")
    filepath_pr = filepath.parent / (filepath.name + "_pr.pdf")

    plotting.density_histogram(elbos_normal, elbos_ood, filepath=filepath_hist, neptune_run=neptune_run)
    plotting.density_plot(elbos_normal, elbos_ood, filepath=filepath_dens, neptune_run=neptune_run)
    auroc, auprc = plotting.roc_pr_curves(elbos_normal, elbos_ood, filepath_roc=filepath_roc, filepath_pr=filepath_pr,
                                          neptune_run=neptune_run, return_fp_fn=False)

    mean_elbo_normal = np.mean(elbos_normal)
    mean_elbo_ood = np.mean(elbos_ood)

    scores = {"auroc": auroc, "auprc": auprc, "mean_elbo_normal": mean_elbo_normal, "mean_elbo_ood": mean_elbo_ood}
    with open(filepath.parent / (filepath.name + f"_scores.p"), "wb") as f:
        pickle.dump(scores, f)
    # with open(filepath.parent / (filepath.name + f"_scores.json"), "w") as f:
    #     json.dump(scores, f)

    if neptune_run is not None:
        neptune_run[f"ood_scores/{filepath.name}_auroc"] = auroc
        neptune_run[f"ood_scores/{filepath.name}_auprc"] = auprc
        neptune_run[f"ood_scores/{filepath.name}_mean_elbo_normal"] = mean_elbo_normal
        neptune_run[f"ood_scores/{filepath.name}_mean_elbo_ood"] = mean_elbo_ood
