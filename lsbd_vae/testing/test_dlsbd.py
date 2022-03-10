import numpy as np
import matplotlib.pyplot as plt
from lsbd_vae.metrics import dlsbd_metric


def test_hypertorus_perfect_embedding(n_subgroups, num_angles=20):
    k = [1] * n_subgroups
    num_angles = num_angles
    k_values = dlsbd_metric.create_combinations_k_values_range(-2, 2, n_subgroups)
    angles_regular = [dlsbd_metric.get_regular_angles(num_angles) for num_transform in range(n_subgroups)]
    angles_combinations = np.array(np.meshgrid(*angles_regular, indexing="ij"))
    angles_flat = angles_combinations.reshape(n_subgroups, -1)
    perfect_embeddings = np.concatenate(tuple([[np.cos(angles), np.sin(angles)] for angles in angles_combinations]),
                                        axis=0)
    z = np.moveaxis(perfect_embeddings, 0, -1)
    perfect_embeddings_flat = z.reshape((-1, 2 * n_subgroups))
    output = dlsbd_metric.dlsbd_torus(z, k_values, verbose=1)
    print(output)
    assert output[0] < np.finfo(float).eps, "Metric is not zero for perfect embedding"
    return 0



if __name__ == "__main__":
    # test_train_ulsbd_vae()
    test_hypertorus_perfect_embedding(2)
    test_hypertorus_perfect_embedding(3)
    print("Everything passed")
