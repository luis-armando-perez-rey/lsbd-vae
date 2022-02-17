import os
import sys
import numpy as np
from typing import List, Optional, Dict

sys.path.append(os.getcwd())
from lsbd_vae.models.latentspace import LatentSpace, GaussianLatentSpace, HyperSphericalLatentSpace

LATENT_SPACE_MAP = {
    "e": GaussianLatentSpace,
    "s": HyperSphericalLatentSpace
}


def get_ls_list(latent_types: List[str], latent_dims: List[int], kl_weights: Optional[List[float]],
                kwargs_list: Optional[List[Dict]] = None) -> \
        List[LatentSpace]:
    """
    Creates a list of LatentSpace elements from a List of latent_types, latent_dims and kl_weights
    Args:

        latent_types: List of latent space types as strings, see LATENT_SPACE_MAP
        latent_dims: List of latent dims
        kl_weights: List of KL weights
        kwargs_list: List of kwargs per latent space if None, no

    Returns:

    """
    if kl_weights is None:
        kl_weights = np.ones(len(latent_types), dtype=np.float)
    if kwargs_list is None:
        kwargs_list = [{}] * len(latent_types)
    assert len(latent_types) == len(latent_dims) == len(
        kl_weights) == len(kwargs_list), "Latent types are not the same length as latent dims and kl weights"
    list_latent = []
    for num_latent, (latent_type, latent_dim, kl_weight) in enumerate(zip(latent_types, latent_dims, kl_weights)):
        assert latent_type in LATENT_SPACE_MAP.keys(), f"Latent type {latent_type} not in available latent " \
                                                       f"spaces keys {LATENT_SPACE_MAP.keys()} "
        list_latent.append(
            LATENT_SPACE_MAP[latent_type](dim=latent_dim, kl_weight=kl_weight, **kwargs_list[num_latent]))
    return list_latent
