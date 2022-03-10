import numpy as np
import matplotlib.pyplot as plt
from lsbd_vae.metrics import dlsbd_metric


def get_perfect_embeddings(n_subgroups: int, n_angles: int):
    """
    Produces embeddings on the hypertorus with n_subgroups S^1 circles with points uniformly spaced on the circles
    :param n_subgroups: number of circles in the hypertorus
    :param n_angles: number of angles on each circle
    :return:
    """
    angles_regular = [dlsbd_metric.get_regular_angles(n_angles) for num_transform in range(n_subgroups)]
    angles_combinations = np.array(np.meshgrid(*angles_regular, indexing="ij"))
    angles_flat = angles_combinations.reshape(n_subgroups, -1)
    perfect_embeddings = np.concatenate(tuple([[np.cos(angles), np.sin(angles)] for angles in angles_combinations]),
                                        axis=0)
    perfect_embeddings = np.moveaxis(perfect_embeddings, 0, -1)
    return perfect_embeddings


def test_hypertorus_perfect_embedding(n_subgroups: int, n_angles: int = 20):
    """
    Test measurement of dlsbd metric on perfect embeddings on hypertorus
    :param n_subgroups: number of S^1 circles in the hypertorus
    :param num_angles: number of angles considered on each S^1 circle
    :return:
    """
    k = [1] * n_subgroups
    # Rotation group representation hyperparameter search can be within [-1, 1] for perfect embeddings
    k_values = dlsbd_metric.create_combinations_k_values_range(-1, 1, n_subgroups)
    z = get_perfect_embeddings(n_subgroups, n_angles)
    output = dlsbd_metric.dlsbd(z, k_values, verbose=1, factor_manifold="torus")
    print(output)
    assert output[0] < np.finfo(float).eps, "Metric is not zero for perfect embedding"
    return 0


def test_hypercylinder_perfect_embedding(n_subgroups: int, n_angles: int = 20, num_objects: int = 5):
    """
    Test measurement of dlsbd metric on perfect embeddings on hypercylinder
    :param n_subgroups: number of S^1 circles in the hypertorus
    :param num_angles: number of angles considered on each S^1 circle
    :return:
    """
    k = [1] * n_subgroups
    # Rotation group representation hyperparameter search can be within [-1, 1] for perfect embeddings
    k_values = dlsbd_metric.create_combinations_k_values_range(-1, 1, n_subgroups)
    z = get_perfect_embeddings(n_subgroups, n_angles)
    z = np.stack([z] * num_objects, axis=0)
    output = dlsbd_metric.dlsbd(z, k_values, verbose=1, factor_manifold="cylinder")
    print(output)
    assert output[0] < np.finfo(float).eps, "Metric is not zero for perfect embedding"
    return 0

if __name__ == "__main__":
    # test_train_ulsbd_vae()
    test_hypertorus_perfect_embedding(2)
    test_hypertorus_perfect_embedding(3)
    test_hypercylinder_perfect_embedding(2)
    test_hypercylinder_perfect_embedding(3)
    print("Everything passed")
