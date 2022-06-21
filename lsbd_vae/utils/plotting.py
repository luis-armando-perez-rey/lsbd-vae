import numpy as np
from typing import List, Optional, Tuple
from matplotlib import patches, pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA


def plot_subset(x_array, cols=None, outlines=True) -> Figure:
    """ Input: matrix of images of shape: (rows, cols, h, w, d)
    """
    x_rows, x_cols, height, width, depth = x_array.shape
    assert depth == 1 or depth == 3, "x_array must contain greyscale or RGB images"
    cols = (cols if cols else x_cols)
    rows = x_rows * int(np.ceil(x_cols / cols))

    fig = plt.figure(figsize=(cols * 2, rows * 2))

    def draw_subplot(x_, ax_):
        if depth == 1:
            plt.imshow(x_.reshape([height, width]), cmap="Greys_r", vmin=0, vmax=1)
        elif depth == 3:
            plt.imshow(x_)
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for j, x_row in enumerate(x_array):
        for i, x in enumerate(x_row[:x_cols], 1):
            # display original
            ax = plt.subplot(rows, cols, i + j * cols * (rows / len(x_array)))  # rows, cols, subplot numbered from 1
            draw_subplot(x, ax)

    return fig


def plot_histograms(s_array, cols=None, outlines=True, filepath=None):
    """ Input: array of categorical variables of shape: (n_variables, n_classes) """
    n_variables, n_classes = s_array.shape
    cols = (cols if cols else n_variables)
    rows = int(np.ceil(n_variables / cols))

    plt.figure(figsize=(cols * 2, rows * 2))

    def draw_subplot(s_, ax_):
        plt.bar(range(n_classes), s_)
        # ax_.set_ylim(0, 1)  # this doesn't work well for a large number of classes, better not fix the y-axis
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for i, s in enumerate(s_array, 1):
        # display original
        ax = plt.subplot(rows, cols, i)  # rows, cols, subplot numbered from 1
        draw_subplot(s, ax)

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath + ".png", bbox_inches='tight')
    plt.close()


def plot_rotations(rotations_array, n_cols=5, filepath=None):
    assert len(rotations_array.shape) == 3, "rotations_array must have shape (sample_size, n_rotations, 2)"
    sample_size = int(rotations_array.shape[0])
    n_rotations = int(rotations_array.shape[1])
    n_rows = np.ceil(sample_size / n_cols)

    for i, rotations in enumerate(rotations_array):
        x = rotations[:, 0]
        y = rotations[:, 1]
        c = np.arange(n_rotations)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.axis("off")
        plt.axis("equal")
        plt.scatter(x, y, c=c, cmap="hsv", marker=".")

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath + ".png", bbox_inches='tight')
    plt.close()


def plot_manifold_2d(decoding_function, grid_x, grid_y):
    """display a 2D manifold"""
    # grid_x = np.array of shape (n_x,)
    # grid_y = np.array of shape (n_y,)
    # decoding_function should take as input np.array of shape (batch_size, 2) and produce an image
    #   where batch_size should be n_x
    n_x = len(grid_x)
    grid_x_reshape = np.expand_dims(grid_x, axis=1)  # shape (n_x, 1)
    x_array = []
    for j, y in enumerate(grid_y):
        y_rep = np.repeat(y, n_x)  # shape (n_x,)
        y_rep = np.expand_dims(y_rep, axis=1)  # shape (n_x, 1)
        z_sample = np.concatenate((grid_x_reshape, y_rep), axis=1)  # shape (n_x, 2), suitable for decoder
        x_decoded = decoding_function(z_sample)  # shape (n_x, height, width, depth)
        x_array.append(x_decoded)
    x_array = np.array(x_array)

    plot_subset(x_array)


def yiq_to_rgb(yiq):
    """
    Convert YIQ colors to RGB.
    :param yiq: yiq colors, shape (n_samples, 3)
    :return:
    """
    conv_matrix = np.array([[1., 0.956, 0.619],
                            [1., -0.272, 0.647],
                            [1., -1.106, 1.703]])
    return np.tensordot(yiq, conv_matrix, axes=((-1,), (-1)))


def yiq_embedding(theta, phi):
    """
    Embed theta and phi into a YIQ color space.
    :param theta: Theta angle in radians
    :param phi: Phi angle in radians
    :return:
    """
    result = np.zeros(theta.shape + (3,))
    steps = 12
    rounding = True
    if rounding:
        theta_rounded = 2 * np.pi * np.round(steps * theta / (2 * np.pi)) / steps
        phi_rounded = 2 * np.pi * np.round(steps * phi / (2 * np.pi)) / steps
        theta = theta_rounded
        phi = phi_rounded
    result[..., 0] = 0.5 + 0.14 * np.cos((theta + phi) * steps / 2) - 0.2 * np.sin(phi)
    result[..., 1] = 0.25 * np.cos(phi)
    result[..., 2] = 0.25 * np.sin(phi)
    return yiq_to_rgb(result)


def plot_training_output(training_output, filepath=None):
    for metric, values_list in training_output.items():
        # replace +/- inf values to NaN, so nanquantile ignores them, to prevent inf axes
        values_list = np.array(values_list)
        values_list[values_list == np.inf] = np.NaN
        values_list[values_list == -np.inf] = np.NaN
        gap = 0.05
        bottom = np.nanquantile(values_list, gap)
        top = np.nanquantile(values_list, 1 - gap)
        bottom -= gap * (top - bottom)
        top += gap * (top - bottom)
        plt.ylim(bottom, top)
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.plot(values_list)
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath + "_" + metric + ".png", bbox_inches='tight')
        plt.close()


def plot_torus_angles(encoded, colors) -> Figure:
    """
    Plot the torus angles in the latent space. The torus angles are the angles of the torus surface.
    :param encoded: Encoded latent space embedding.
    :param colors: Colors of the points.
    :return:
    """
    encoded_horizontal_angle = np.arctan2(encoded[1][:, :, 0], encoded[1][:, :, 1])
    encoded_vertical_angle = np.arctan2(encoded[4][:, :, 0], encoded[4][:, :, 1])
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.scatter(encoded_horizontal_angle, encoded_vertical_angle, color=colors)
    ax.set_title("Torus encoded")
    return fig


def plot_decoded(decoded, height_grid, width_grid, divisor=2) -> Figure:
    fig = plt.figure(figsize=(5, 5))
    for data_num in range(height_grid // divisor * width_grid // divisor):
        ax = fig.add_subplot(height_grid // divisor, width_grid // divisor, data_num + 1)
        if decoded.shape[-1] == 1:
            ax.imshow(decoded[data_num * divisor, 0, :, :, 0])
        else:
            ax.imshow(decoded[data_num * divisor, 0, :, :, :])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    return fig


def plot_euclidean_embedding(encoded, colors) -> Figure:
    """
    Plot the Euclidean embedding of the latent space.
    :param encoded: encoded data points
    :param colors: colors to be used for the plotted embeddings
    :return: figure output
    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(encoded[0][:, :, 0], encoded[0][:, :, 1], color=colors)
    ax.add_artist(patches.Circle((0, 0), 1.0, fill=False, zorder=-1))
    ax.set_title("Euclidean embedding 1")
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(encoded[3][:, :, 0], encoded[3][:, :, 1], color=colors)
    ax.add_artist(patches.Circle((0, 0), 1.0, fill=False, zorder=-1))
    ax.set_title("Euclidean embedding 2")
    return fig

def plot_latent_dimension_combinations(z, colors_flat) -> Tuple[Figure, List[Axis]]:
    """
    Plotting of the embeddings for all possible pair-wise combinations of their dimensions
    Args:
        z: Array with num_vectors embeddings of z_dim dimensions (num_vectors, z_dim)
        colors_flat: Array of color values for each of the num_vectors embeddings (num_vectors, color_channels)

    Returns:
        fig, axes
    """
    total_dimensions = z.shape[-1]
    fig, axes = plt.subplots(total_dimensions, total_dimensions, figsize=(10, 10))
    for dim1 in range(total_dimensions):
        for dim2 in range(total_dimensions):
            axes[dim1, dim2].scatter(z[:, dim1], z[:, dim2], c=colors_flat)
            axes[dim1, dim2].set_title("Dim ({}, {})".format(dim1, dim2))
    return fig, axes


def angle_to_point(angles):
    """" Takes an array of angles between 0 and 2*Pi (with shape (...,))
         and computes an array of points on the 2D unit circle corresponding to those angles (shape (..., 2)) """
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def latent_traversals_s1_x_rd(decoder, z2_dim, n_samples=10, n_traversals=10):
    """
    Plot latent traversals for a latent space S^1 x R^d, randomly sampling n_samples from a Gaussian prior on R^d,
        and traversing over S^1 in n_traversals steps
    Args:
        decoder (tf.keras.Model): decoder model with two inputs of dim 2 and d, respectively, for S^1 and R^d
        z2_dim (int): dimension d of R^d
        n_samples (int): number of samples from R^d, and number of rows in plot
        n_traversals (int): number of traversals in S^1, and number of columns in plot

    Returns:
         fig
    """
    angles_grid = np.linspace(0, 2 * np.pi, num=n_traversals, endpoint=False)
    z1_grid = angle_to_point(angles_grid)  # shape (n_traversals, 2)
    z2_samples = np.random.normal(size=(n_samples, z2_dim))
    x_array = np.stack(
        [decoder.predict([z1_grid, np.tile(z, (n_traversals, 1))])
         for z in z2_samples],
        axis=0)  # shape (n_samples, n_traversals, h, w, d)
    return plot_subset(x_array)


def plot_grid_images(images: np.array, figsize=None):
    """
    Plot images with shape (n_objects, n_views, *image_shape) into a grid with no ax ticks
    Args:
        images: array with shape (n_objects, n_views, *image_shape)
        figsize: figure size
    Returns:
        returns matplotlib figure and axis
    """

    num_objects = images.shape[0]
    num_views = images.shape[1]
    if figsize is None:
        fig, axes = plt.subplots(num_objects, num_views, figsize=(num_views, num_objects))
    else:
        fig, axes = plt.subplots(num_objects, num_views, figsize=figsize)
    if num_objects == 1:
        for num_view, image in enumerate(images[0]):
            axes[num_view].imshow(image)
            axes[num_view].set_xticks([])
            axes[num_view].set_yticks([])
    else:
        for num_object, views in enumerate(images):
            for num_view, image in enumerate(views):
                axes[num_object, num_view].imshow(image)
                axes[num_object, num_view].set_xticks([])
                axes[num_object, num_view].set_yticks([])
    return fig, axes


def plot_objects_pca(object_images, embeddings, zoom=1.0, ax=None):
    """
    Plot images on the same location as the embeddings with certain zoom
    Args:
        object_images: images of the objects to be plotted
        zoom: zoom factor for the plotted images
        embeddings: embeddings to be projected by PCA
        ax: axis to plot on if None, a new figure is created

    Returns:

    """
    # Plot embeddings
    if embeddings.shape[-1] == 2:
        print("PCA: Embeddings are already 2 dimensional, no PCA applied")
        x_embedded = embeddings
    else:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        x_embedded = pca.transform(embeddings)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = plt.gcf()
    for num_embedding, embedding in enumerate(x_embedded):
        image_scatter(embedding[0], embedding[1], object_images[num_embedding], ax=ax, zoom=zoom)
    ax.set_title("PCA object embeddings")
    return fig, ax


def image_scatter(x, y, image, ax=None, zoom=1.0):
    """
    Plot an image on a scatter plot with a certain zoom factor positioned at (x, y)
    :param x: positions in the x axis of the image
    :param y: positions in the y axis of the image
    :param image: image to be plotted
    :param ax: axis to plot on if None, a new figure is created
    :param zoom: zoom of the plotted image
    :return:
    """
    if ax is None:
        ax = plt.gca()

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_data_examples_tworows(num_images, data_class, cmap: Optional[str] = None):
    """
    Plot a subset of images from a data class into two rows showing the factors changing in the dataset
    :param num_images: number of images to show in each row
    :param data_class: data class to plot from
    :param cmap: optional cmap to use for the images, recommended use binary_r for pixel datasets
    :return:
    """
    fig, axes = plt.subplots(2, num_images, figsize=(8, 2))
    factor_shape = data_class.images.shape[:2]
    image_indexes0 = np.arange(0, factor_shape[0], factor_shape[0] // num_images)
    image_indexes1 = np.arange(0, factor_shape[1], factor_shape[1] // num_images)
    for num_ax in range(num_images):
        if cmap is None:
            axes[0, num_ax].imshow(data_class.images[image_indexes0[num_ax], 0])
            axes[1, num_ax].imshow(data_class.images[0, image_indexes1[num_ax]])
        else:
            axes[0, num_ax].imshow(data_class.images[image_indexes0[num_ax], 0], cmap=cmap)
            axes[1, num_ax].imshow(data_class.images[0, image_indexes1[num_ax]], cmap=cmap)
        axes[0, num_ax].set_xticks([])
        axes[0, num_ax].set_yticks([])
        axes[1, num_ax].set_xticks([])
        axes[1, num_ax].set_yticks([])
    return fig, axes


def plot_data_examples_grid(num_images, data_class, cmap: Optional[str] = None) -> Tuple[Figure, Axis]:
    fig = plt.figure(figsize=(10, 10))
    for num_i, i in enumerate(
            np.clip(np.arange(0, len(data_class.images) + 1, (len(data_class.images)) // (num_images - 1)), 0,
                    len(data_class.images))):
        for num_j, j in enumerate(
                np.clip(np.arange(0, data_class.images.shape[1] + 1, (data_class.images.shape[1]) // (num_images - 1)),
                        0, data_class.images.shape[1] - 1)):
            ax = plt.subplot2grid((num_images, num_images), (num_i, num_j))
            if cmap is None:
                ax.imshow(data_class.images[i, j, ...])
            else:
                ax.imshow(data_class.images[i, j, ...], cmap="binary_r")
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
    # noinspection PyUnboundLocalVariable
    return fig, ax
