import numpy as np


def select_factor_combinations(factor_values, factor_ranges):
    """
    Given an array of factor values, and a range of values to select for each factor,
    returns the indices for which the factor values fall within all those ranges,
    as well as the remaining indices where at least one factor value is outside the given range.

    Args:
        factor_values (np.array): array of shape (n_datapoints, n_factors) containing the factor values data
        factor_ranges (tuple): n_factors-tuple of 2-tuples, each representing the (min, max)-range for a factor
    """
    n_factors = len(factor_ranges)
    assert len(factor_values.shape) == 2, "factor_values must have shape (n_datapoints, n_factors)"
    assert factor_values.shape[1] == n_factors, "n_factors in factor_values and factor_ranges don't match"

    truefalse_global = None  # just to silent "might be referenced before assignment" warning
    for i in range(n_factors):
        assert len(factor_ranges[i]) == 2, "elements of factor_ranges must be 2-tuples of (min, max) values"
        truefalse_current = np.logical_and(factor_values[:, i] >= factor_ranges[i][0],
                                           factor_values[:, i] < factor_ranges[i][1])
        if i == 0:
            truefalse_global = truefalse_current
        else:
            truefalse_global = np.logical_and(truefalse_global, truefalse_current)

    indices_selected = np.squeeze(np.argwhere(truefalse_global), axis=1)
    indices_unselected = np.squeeze(np.argwhere(np.logical_not(truefalse_global)), axis=1)

    return indices_selected, indices_unselected


def setup_circles_dataset_labelled_pairs(images, factor_values, n_labels, max_factor_values=None):
    """
    Args:
        images (np.array): data array of shape (n_datapoints, *input_dim)
        factor_values (np.array): array of factor values, shape (n_datapoints, n_factors)
        n_labels: Number of labelled pairs to generate
        max_factor_values (np.array): array-like of shape (n_factors), containing the max value for each factor
            (assuming min value is 0), all factor values will be scaled w.r.t. this value to be between 0 and 2*Pi
    Returns:
        x_l: Labeled pairs array with shape (n_labels, 2, height, width, depth)
        x_l_transformations: List of length n_factors, each element is an array of shape (n_labels, 2, 1)
            where [:, 0, :] represents the identity transformations,
            and [:, 1, :] represents the transformation from the first to the second element of a pair,
            given as an angle on the unit circle
        x_u: Unlabeled data points with shape (n_data_points - 2*n_labels, 1, height, width, depth)
    """
    # labelling procedure: randomly select n_labels pairs, such that each data point is part of at most one pair.
    # produce the transformation label for each of those pairs.
    n_datapoints = len(images)
    assert len(factor_values) == n_datapoints, "len(images) and len(factor_values) must be equal"
    assert 2 * n_labels <= n_datapoints, "for this procedure 2 * n_labels cannot exceed the number of data points"
    n_factors = factor_values.shape[1]
    if max_factor_values is not None:
        assert len(max_factor_values) == n_factors, "n_factors in factor_values and max_factor_values is not equal"
    else:
        max_factor_values = n_factors * [2 * np.pi]

    # sample 2*n_labels indices, for the data points/pairs to be labelled
    indices = np.random.choice(n_datapoints, size=2 * n_labels, replace=False)
    # split in two halves, for the first and second elements of the pairs
    ind1 = indices[:n_labels]
    ind2 = indices[n_labels:]

    x_l_transformations = []
    for factor_num in range(n_factors):
        differences = (factor_values[ind2, factor_num] - factor_values[ind1, factor_num]) % \
                      max_factor_values[factor_num]
        angles = np.expand_dims(2 * np.pi * differences / max_factor_values[factor_num], axis=1)
        identity_transformations = np.zeros_like(angles)
        x_l_transformations.append(np.stack([identity_transformations, angles], axis=1))

    # set up the set x_l of labelled data points, with shape (n_labels, 2, height, width, depth)
    x1 = images[ind1]  # shape (n_labels, height, width, depth)
    x2 = images[ind2]  # shape (n_labels, height, width, depth)
    x_l = np.stack([x1, x2], axis=1)  # shape (n_labels, 2, height, width, depth)

    # select all remaining data points for the unlabelled set x_u,
    #   with shape (n_unlabelled, 1, height, width, depth)
    mask = np.ones(n_datapoints, dtype=bool)
    mask[indices] = False
    x_u = images[mask]
    x_u = np.expand_dims(x_u, axis=1)  # shape (n_data_points - 2*n_labels, 1, height, width, depth)
    x_l_transformations = tuple(x_l_transformations)
    return x_l, x_l_transformations, x_u


def setup_circles_dataset_labelled_batches(images, factor_values, n_batches, batch_size, max_factor_values=None):
    """
    Args:
        images (np.array): data array of shape (n_datapoints, *input_dim)
        factor_values (np.array): array of factor values, shape (n_datapoints, n_factors)
        n_batches: Number of labelled batches to generate
        batch_size: size of each labelled batch
        max_factor_values (np.array): array-like of shape (n_factors), containing the max value for each factor
            (assuming min value is 0), all factor values will be scaled w.r.t. this value to be between 0 and 2*Pi
    Returns:
        x_l: Labeled pairs array with shape (n_labels, 2, height, width, depth)
        x_l_transformations: List of length n_factors, each element is an array of shape (n_labels, 2, 1)
            where [:, 0, :] represents the identity transformations,
            and [:, 1, :] represents the transformation from the first to the second element of a pair,
            given as an angle on the unit circle
        x_u: Unlabeled data points with shape (n_data_points - 2*n_labels, 1, height, width, depth)
    """
    # labelling procedure: randomly select n_labels batches of size batch_size,
    #   such that each data point is part of at most one batch.
    # produce the transformation label for each of those batches.
    n_datapoints = len(images)
    assert len(factor_values) == n_datapoints, "len(images) and len(factor_values) must be equal"
    assert batch_size * n_batches <= n_datapoints,\
        "for this procedure batch_size * n_labels cannot exceed the number of data points"
    n_factors = factor_values.shape[1]
    if max_factor_values is not None:
        assert len(max_factor_values) == n_factors, "n_factors in factor_values and max_factor_values is not equal"
    else:
        max_factor_values = n_factors * [2 * np.pi]

    # sample batch_size*n_labels indices, for the data points/pairs to be labelled
    indices = np.random.choice(n_datapoints, size=batch_size * n_batches, replace=False)
    # split in batch_size parts, for the consecutive elements of the batches
    indices_partitioned = [indices[i * n_batches: (i+1) * n_batches] for i in range(batch_size)]

    x_l_transformations = []
    for factor_num in range(n_factors):
        angles_factor = []
        for batch_pos in range(batch_size):
            differences = (factor_values[indices_partitioned[batch_pos], factor_num]
                           - factor_values[indices_partitioned[0], factor_num]) \
                          % max_factor_values[factor_num]  # shape (n_batches)
            angles = np.expand_dims(2 * np.pi * differences / max_factor_values[factor_num], axis=1) # (n_batches, 1)
            angles_factor.append(angles)
        transformations_factor = np.stack(angles_factor, axis=1)  # shape (n_batches, batch_size, 1)
        x_l_transformations.append(transformations_factor)  # length n_factors after for-loop

    # set up the set x_l of labelled data points, with shape (n_batches, batch_size, height, width, depth)
    x_l_lst = [images[ind] for ind in indices_partitioned]  # length batch_size, elements of shape (n_batches, h, w, d)
    x_l = np.stack(x_l_lst, axis=1)  # shape (n_batches, batch_size, height, width, depth)

    # select all remaining data points for the unlabelled set x_u,
    #   with shape (n_unlabelled, 1, height, width, depth)
    mask = np.ones(n_datapoints, dtype=bool)
    mask[indices] = False
    x_u = images[mask]
    x_u = np.expand_dims(x_u, axis=1)  # shape (n_data_points - batch_size*n_batches, 1, height, width, depth)
    x_l_transformations = tuple(x_l_transformations)
    return x_l, x_l_transformations, x_u


def dsprites_symmetry_correction(dataset_class):
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
    return factor_values_as_angles_grid, factor_values, factor_values_as_angles


def split_up_data_ood(dataset_class, data_parameters, correct_dsprites_symmetries, use_angles_for_selection,
                      factor_ranges):
    images = dataset_class.flat_images

    # regular factor values are needed to select factor combinations given factor_ranges,
    # factor values as angles are needed for LSBD-VAE training
    if correct_dsprites_symmetries and data_parameters["data"] == "dsprites":
        factor_values_as_angles_grid, factor_values, factor_values_as_angles = \
            dsprites_symmetry_correction(dataset_class)
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
        factor_values_as_angles_ood = None
    else:
        if use_angles_for_selection:
            indices_ood, indices_train = \
                select_factor_combinations(factor_values_as_angles, factor_ranges)
        else:
            indices_ood, indices_train = select_factor_combinations(factor_values, factor_ranges)
        images_train = images[indices_train]
        factor_values_as_angles_train = factor_values_as_angles[indices_train]
        images_ood = images[indices_ood]
        factor_values_as_angles_ood = factor_values_as_angles[indices_ood]

    return images, images_train, images_ood,\
        factor_values_as_angles, factor_values_as_angles_train, factor_values_as_angles_ood,\
        factor_values_as_angles_grid
