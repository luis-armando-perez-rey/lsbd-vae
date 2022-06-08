import numpy as np
import pickle
import json

from lsbd_vae.metrics import dlsbd_metric

from ood_generalisation.modules import plotting, data_selection


def plot_reconstructions(lsbd, x, filepath, neptune_run=None):
    """
    Args:
        lsbd: (un)supervised LSBDVAE class instance
        x: input images array, shape (n_samples, *input_shape)
        filepath: Path object with filepath (excluding extension) where to save image.
        neptune_run: Neptune.ai run
    """
    filepath = filepath.parent / (filepath.name + ".png")
    x_recon = lsbd.reconstruct_images(x, return_latents=False)
    x_array = np.stack([x, x_recon], axis=0)
    plotting.plot_images_grid(x_array, filepath=filepath, neptune_run=neptune_run)


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


def plot_2d_torus_embedding(images_grid, factor_values_as_angles_grid, lsbd, filepath, neptune_run=None,
                            x_dim=0, y_dim=1, factor_ranges=None):
    """
    F = number of factors (and number of latent spaces)

    Args:
        images_grid: shape (n1, ..., nF, h, w, d)
        factor_values_as_angles_grid:  shape (n1, ..., nF, F), given as angles
        lsbd: BaseLSBDVAE instance or subclass
        filepath:
        neptune_run:
        x_dim: value in [0, F), distinct from y_dim
        y_dim: value in [0, F), distinct from x_dim
        factor_ranges: if not None, darken the embeddings of datapoints within the given factor_ranges
    """
    assert x_dim != y_dim, "first and second dimensions should not be the same"
    assert x_dim < lsbd.n_latent_spaces and y_dim < lsbd.n_latent_spaces

    for factor, factor_size in enumerate(factor_values_as_angles_grid.shape[:-1]):
        if factor != x_dim and factor != y_dim:
            random_factor_index = np.random.randint(factor_size)
            # take single element for this factor index, [] brackets ensure that dimension is kept (with size 1)
            images_grid = np.take(images_grid, [random_factor_index], axis=factor)  # size of dim factor is changed to 1
            factor_values_as_angles_grid = np.take(factor_values_as_angles_grid, [random_factor_index], axis=factor)
    flat_images = np.reshape(images_grid, (-1, *images_grid.shape[-3:]))
    flat_factor_mesh = np.reshape(factor_values_as_angles_grid, (-1, factor_values_as_angles_grid.shape[-1]))

    h_angle = flat_factor_mesh[:, x_dim]
    v_angle = flat_factor_mesh[:, y_dim]
    colors = plotting.yiq_embedding(v_angle, h_angle)

    # if factor_ranges are given, give darker shade to the train data, thus highlighting the OOD data
    if factor_ranges is not None:
        indices_ood, indices_train = data_selection.select_factor_combinations(flat_factor_mesh, factor_ranges)
        colors[indices_train] = colors[indices_train] * 0.6

    encodings_list = lsbd.encode_images(flat_images)
    # encoded list is a list of length n_latent_spaces, each item is an array of shape (n, ls_latent_dim)

    encoded_horizontal_angle = np.arctan2(encodings_list[x_dim][:, 1], encodings_list[x_dim][:, 0])
    encoded_vertical_angle = np.arctan2(encodings_list[y_dim][:, 1], encodings_list[y_dim][:, 0])
    plotting.plot_torus_angles(encoded_horizontal_angle, encoded_vertical_angle, colors,
                               filepath=filepath, neptune_run=neptune_run)


def plot_2d_latent_traverals_torus(lsbd, n_gridpoints, filepath, neptune_run=None, x_dim=0, y_dim=1):
    assert x_dim != y_dim, "first and second dimensions should not be the same"
    assert x_dim < lsbd.n_latent_spaces and y_dim < lsbd.n_latent_spaces
    # linear spaces between -pi and pi, shifted so the difference between first and last point is the same as all other
    #   differences modulo 2pi. angles_x correspond to cols, angles_y to rows (following imshow convention, not matrix
    #   convention, to match the embedding plots), angles_y given in reversed order (again to follow imshow convention)
    halfstep = np.pi / n_gridpoints
    angles_x = np.linspace(-np.pi + halfstep, np.pi - halfstep, num=n_gridpoints, endpoint=True)
    angles_y = np.linspace(np.pi - halfstep, -np.pi + halfstep, num=n_gridpoints, endpoint=True)
    # transform to (n_gridpoints, 2) arrays of corresponding values on the unit circle
    grid_x = np.stack([np.cos(angles_x), np.sin(angles_x)], axis=1)
    grid_y = np.stack([np.cos(angles_y), np.sin(angles_y)], axis=1)
    grid_x = np.expand_dims(grid_x, axis=0)  # shape (1, n_gridpoints, 2)
    grid_y = np.expand_dims(grid_y, axis=1)  # shape (n_gridpoints, 1, 2)
    grid_x = np.tile(grid_x, (n_gridpoints, 1, 1))  # shape (n_gridpoints, n_gridpoints, 2)
    grid_y = np.tile(grid_y, (1, n_gridpoints, 1))  # shape (n_gridpoints, n_gridpoints, 2)
    grid_x_flat = np.reshape(grid_x, (n_gridpoints * n_gridpoints, 2))
    grid_y_flat = np.reshape(grid_y, (n_gridpoints * n_gridpoints, 2))
    grid_flat = []
    for i in range(lsbd.n_latent_spaces):
        if i == x_dim:
            grid_flat.append(grid_x_flat)
        elif i == y_dim:
            grid_flat.append(grid_y_flat)
        else:  # fix random value for entire grid
            angle = np.random.uniform(0, 2*np.pi)
            circlepoint = np.array([np.cos(angle), np.sin(angle)])
            circlepoint = np.expand_dims(circlepoint, axis=0)  # shape (1, 2)
            circlepoint = np.tile(circlepoint, (n_gridpoints * n_gridpoints, 1))  # shape (n_gridpoints**2, 2)
            grid_flat.append(circlepoint)
    # decode
    x_generated = lsbd.decode_latents(grid_flat)  # shape (n_gridpoints*n_gridpoints, *input_dim)
    x_generated = np.reshape(x_generated, (n_gridpoints, n_gridpoints, *x_generated.shape[1:]))
    # plot output
    plotting.plot_images_grid(x_generated, filepath=filepath, neptune_run=neptune_run)


def ood_detection(lsbd, x_normal, x_ood, filepath, neptune_run=None):
    reconstruction_losses_normal, kl_losses_normal, elbos_normal = lsbd.compute_losses_and_elbos(x_normal)
    reconstruction_losses_ood, kl_losses_ood, elbos_ood = lsbd.compute_losses_and_elbos(x_ood)

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


def compute_d_lsbd(lsbd, images_grid, n_factors, filepath, neptune_run=None):
    k_values = dlsbd_metric.create_combinations_k_values_range(-2, 2, n_transforms=n_factors)
    flat_images = np.reshape(images_grid, (-1, *images_grid.shape[-3:]))  # works for images with n_data_dims=3
    encodings_list = lsbd.encode_images(flat_images)  # n_latent_spaces lists of shape (n_images, ls_latent_dim) each
    encodings_flat = np.concatenate(encodings_list, axis=1)  # array of shape (n_images, total_latent_dim)
    encodings_grid = np.reshape(encodings_flat, (*images_grid.shape[:-3], encodings_flat.shape[1]))
    score, k_min = dlsbd_metric.dlsbd(encodings_grid, k_values, factor_manifold="torus")

    with open(filepath.parent / (filepath.name + ".p"), "wb") as f:
        pickle.dump(score, f)

    if neptune_run is not None:
        neptune_run[f"disentanglement_metrics/{filepath.name}"] = score