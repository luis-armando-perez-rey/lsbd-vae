import numpy as np


def select_factor_combinations(factor_values, factor_ranges):
    """
    Given a array of factor values, and a range of values to select for each factor,
    returns the indices for which the factor values fall within all those ranges,
    as well as the remaining indices where at least one factor value is outside of the given range.

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
                                           factor_values[:, i] <= factor_ranges[i][1])
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
