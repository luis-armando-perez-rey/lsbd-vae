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


def plot_2d_torus_embedding(dataset_class, lsbd, filepath, neptune_run=None):
    encodings_list = lsbd.encode_images(dataset_class.flat_images)
    # encoded list is a list of length n_latent_spaces, each item is an array of shape (n, ls_latent_dim)

    v_angle = dataset_class.flat_factor_mesh_as_angles[:, 0]
    h_angle = dataset_class.flat_factor_mesh_as_angles[:, 1]
    colors = plotting.yiq_embedding(v_angle, h_angle)

    encoded_horizontal_angle = np.arctan2(encodings_list[0][:, 0], encodings_list[0][:, 1])
    encoded_vertical_angle = np.arctan2(encodings_list[1][:, 0], encodings_list[1][:, 1])
    plotting.plot_torus_angles(encoded_horizontal_angle, encoded_vertical_angle, colors,
                               filepath=filepath, neptune_run=neptune_run)


def plot_2d_latent_traverals_torus(lsbd, n_gridpoints, filepath, neptune_run=None):
    # linear spaces from 0 to 2*pi (exclusive)
    angles_x = np.linspace(0, 2 * np.pi, num=n_gridpoints, endpoint=False)
    angles_y = np.linspace(0, 2 * np.pi, num=n_gridpoints, endpoint=False)
    # transform to (n_gridpoints, 2) arrays of corresponding values on the unit circle
    grid_x = np.stack([np.cos(angles_x), np.sin(angles_x)], axis=1)
    grid_y = np.stack([np.cos(angles_y), np.sin(angles_y)], axis=1)
    # make (n_x*n_y, 4) array as decoder input
    grid_x = np.expand_dims(grid_x, axis=1)  # shape (n_gridpoints, 1, 2)
    grid_y = np.expand_dims(grid_y, axis=0)  # shape (1, n_gridpoints, 2)
    grid_x = np.tile(grid_x, (1, n_gridpoints, 1))  # shape (n_gridpoints, n_gridpoints, 2)
    grid_y = np.tile(grid_y, (n_gridpoints, 1, 1))  # shape (n_gridpoints, n_gridpoints, 2)
    grid_x_flat = np.reshape(grid_x, (n_gridpoints * n_gridpoints, 2))
    grid_y_flat = np.reshape(grid_y, (n_gridpoints * n_gridpoints, 2))
    grid_flat = [grid_x_flat, grid_y_flat]
    # decode
    x_generated = lsbd.decode_latents(grid_flat)  # shape (n_gridpoints*n_gridpoints, *input_dim)
    x_generated = np.reshape(x_generated, (n_gridpoints, n_gridpoints, *x_generated.shape[1:]))
    # plot output
    plotting.plot_images_grid(x_generated, filepath=filepath, neptune_run=neptune_run)


def ood_detection(x_normal, x_ood, filepath, neptune_run=None):
    elbos_normal = ...
    elbos_ood = ...

    filepath_hist = filepath.parent / (filepath.name + "_hist.pdf")
    filepath_dens = filepath.parent / (filepath.name + "_dens.pdf")
    filepath_roc = filepath.parent / (filepath.name + "_roc.pdf")
    filepath_pr = filepath.parent / (filepath.name + "_pr.pdf")

    plotting.density_histogram(elbos_normal, elbos_ood, filepath=filepath_hist, neptune_run=neptune_run)
    plotting.density_plot(elbos_normal, elbos_ood, filepath=filepath_dens, neptune_run=neptune_run)
    auroc, auprc = plotting.roc_pr_curves(elbos_normal, elbos_ood, filepath_roc=filepath_roc, filepath_pr=filepath_pr,
                                          neptune_run=neptune_run, return_fp_fn=False)

    if neptune_run is not None:
        neptune_run[f"ad_scores/{filepath.name}_auroc"] = auroc
        neptune_run[f"ad_scores/{filepath.name}_auprc"] = auprc
