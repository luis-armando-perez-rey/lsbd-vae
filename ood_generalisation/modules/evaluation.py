import numpy as np

from ood_generalisation.modules import plotting


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


def plot_2d_torus_embedding(images_grid, factor_values_as_angles_grid, lsbd, filepath, neptune_run=None, x_dim=0, y_dim=1):
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

    encodings_list = lsbd.encode_images(flat_images)
    # encoded list is a list of length n_latent_spaces, each item is an array of shape (n, ls_latent_dim)

    v_angle = flat_factor_mesh[:, x_dim]
    h_angle = flat_factor_mesh[:, y_dim]
    colors = plotting.yiq_embedding(v_angle, h_angle)

    encoded_horizontal_angle = np.arctan2(encodings_list[x_dim][:, 0], encodings_list[x_dim][:, 1])
    encoded_vertical_angle = np.arctan2(encodings_list[y_dim][:, 0], encodings_list[y_dim][:, 1])
    plotting.plot_torus_angles(encoded_horizontal_angle, encoded_vertical_angle, colors,
                               filepath=filepath, neptune_run=neptune_run)


def plot_2d_latent_traverals_torus(lsbd, n_gridpoints, filepath, neptune_run=None, x_dim=0, y_dim=1):
    assert x_dim != y_dim, "first and second dimensions should not be the same"
    assert x_dim < lsbd.n_latent_spaces and y_dim < lsbd.n_latent_spaces
    # linear spaces from 0 to 2*pi (exclusive)
    angles_x = np.linspace(0, 2 * np.pi, num=n_gridpoints, endpoint=False)
    angles_y = np.linspace(0, 2 * np.pi, num=n_gridpoints, endpoint=False)
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
    #grid_flat = [grid_x_flat, grid_y_flat]
    # decode
    x_generated = lsbd.decode_latents(grid_flat)  # shape (n_gridpoints*n_gridpoints, *input_dim)
    x_generated = np.reshape(x_generated, (n_gridpoints, n_gridpoints, *x_generated.shape[1:]))
    # plot output
    plotting.plot_images_grid(x_generated, filepath=filepath, neptune_run=neptune_run)


def ood_detection(lsbd, x_normal, x_ood, filepath, neptune_run=None):
    reconstruction_losses_normal, kl_losses_normal, elbos_normal = lsbd.compute_losses_and_elbos(x_normal)
    reconstruction_losses_ood, kl_losses_ood, elbos_ood = lsbd.compute_losses_and_elbos(x_ood)
    elbos_normal = elbos_normal.numpy()
    elbos_ood = elbos_ood.numpy()

    filepath_hist = filepath.parent / (filepath.name + "_hist.pdf")
    filepath_dens = filepath.parent / (filepath.name + "_dens.pdf")
    filepath_roc = filepath.parent / (filepath.name + "_roc.pdf")
    filepath_pr = filepath.parent / (filepath.name + "_pr.pdf")

    plotting.density_histogram(elbos_normal, elbos_ood, filepath=filepath_hist, neptune_run=neptune_run)
    plotting.density_plot(elbos_normal, elbos_ood, filepath=filepath_dens, neptune_run=neptune_run)
    auroc, auprc = plotting.roc_pr_curves(elbos_normal, elbos_ood, filepath_roc=filepath_roc, filepath_pr=filepath_pr,
                                          neptune_run=neptune_run, return_fp_fn=False)

    if neptune_run is not None:
        neptune_run[f"ood_scores/{filepath.name}_auroc"] = auroc
        neptune_run[f"ood_scores/{filepath.name}_auprc"] = auprc
