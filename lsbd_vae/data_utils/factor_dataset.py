from typing import Tuple, List
import numpy as np
from tensorflow.keras.utils import to_categorical


class FactorImageDataset:
    """
    Abstract class for datasets that are a Cartesian product of factors

    Args:
        images: the image data itself, shape (f1, ..., fk, h, w, d)
        factor_values_list: list of 1D arrays, one for each factor, containing all factor values present in the data.
            If None, max_factor_values must be provided, and equal spacing is assumed based on these max values.
        max_factor_values: list of maximum values that indicate scale of factors (minimum is assumed to be 0).
            If None, the number of factors in images is used.
        factor_names: list of strings indicating the meaning of each factor.
            If None, ["factor0", ..., "factor{k-1}"] is used.

    Attributes:
        images (np.ndarray): the image data itself, shape (f1, ..., fk, h, w, d)
        factor_values_list: list of 1D arrays, one for each factor, containing all factor values present in the data
        max_factor_values: list of maximum values that indicate scale of factors (minimum is assumed to be 0)
        factor_names: list of strings indicating the meaning of each factor
    """

    def __init__(self, images: np.ndarray,
                 factor_values_list: List[np.ndarray] = None,
                 max_factor_values: List[float] = None,
                 factor_names: List[str] = None,
                 labels=None,
                 ):
        assert len(images.shape) >= 3, "images must have shape (..., h, w, d)"
        self.images = images

        # setup and/or check factor_values_list and max_factor_values
        if factor_values_list is None:  # if no factors are given, assume they are equally spaced
            if max_factor_values is None:  # if no max values given, assume n_factors as max
                max_factor_values = list(self.factors_shape)
            else:
                assert len(max_factor_values) == self.n_factors, \
                    "max_factor_values length should equal the number of factors"
            factor_values_list = []
            for max_factor_value, n_values in zip(max_factor_values, self.factors_shape):
                factor_values_list.append(np.linspace(0, max_factor_value, num=n_values, endpoint=False))
        else:
            assert max_factor_values is not None, "max_factor_values must be specified if factors are given"
            assert len(max_factor_values) == len(factor_values_list) == self.n_factors, \
                "max_factor_values and factor_values_list must have length equal to n_factors"

        # setup or check factor_names
        if factor_names is None:
            factor_names = [f"factor{i}" for i in range(self.n_factors)]
        else:
            assert len(factor_names) == self.n_factors, "factor_names length should equal the number of factors"

        # set attributes
        self.factor_values_list = factor_values_list
        self.max_factor_values = max_factor_values
        self.factor_names = factor_names

        # set class_labels
        self.labels = labels

    @property
    def factors_shape(self) -> Tuple:
        """
        Returns:
            factors_shape: data shape without the final (h, w, d) dimensions
        """
        return self.images.shape[:-3]

    @property
    def n_factors(self) -> int:
        """
        Returns:
            n_factors: the number of factors in images
        """
        return len(self.factors_shape)

    @property
    def factors_as_angles(self):
        """
        Returns:
            factors_as_angles: factor_values_list rescaled to a scale from 0 to 2pi
        """
        angles = []
        for factor, max_value in zip(self.factor_values_list, self.max_factor_values):
            angles.append(2 * np.pi * factor / max_value)
        return angles

    @property
    def image_shape(self):
        """
        Returns:
            image_shape: shape of the images (height, width, depth)
        """
        return self.images.shape[-3:]

    @property
    def n_data_points(self) -> int:
        """
        Returns:
            n_data_points: number of images in the full dataset
        """
        n_data_points = int(np.prod(self.factors_shape))
        return n_data_points

    @property
    def factor_mesh(self) -> np.ndarray:
        """
        Returns:
            factor mesh: array of shape (n1, n2, n3, ..., n_nfactors, nfactors)
        """
        factor_meshes = np.meshgrid(*self.factor_values_list, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def factor_mesh_enumerated(self) -> np.ndarray:
        """
        Returns:
            factor mesh: array of shape (n1, n2, n3, ..., n_nfactors, nfactors)
        """
        enumerated_factor_list = [np.arange(0, len(self.factor_values_list[num_factor])) for num_factor in
                                  range(self.n_factors)]
        factor_meshes = np.meshgrid(*enumerated_factor_list, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def flat_factor_mesh_enumerated(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: array of shape (n1*n2*n3...*n_nfactors, nfactors)
        """
        return self.factor_mesh_enumerated.reshape((self.n_data_points, self.n_factors))

    @property
    def factor_mesh_as_angles(self) -> np.ndarray:
        """
        Returns:
            factor_mesh_as_angles: same as factor_mesh, but with values scaled from 0 to 2pi
        """
        factor_meshes = np.meshgrid(*self.factors_as_angles, indexing="ij")
        factor_mesh = np.stack(factor_meshes, axis=-1)  # (n1, n2, n3, ..., n_n_factors ,n_factors)
        return factor_mesh

    @property
    def flat_factor_mesh(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: array of shape (n1*n2*n3...*n_nfactors, nfactors)
        """
        return self.factor_mesh.reshape((self.n_data_points, self.n_factors))

    @property
    def flat_factor_mesh_as_angles(self) -> np.ndarray:
        """
        Returns:
            flat_factor_mesh: same as flat_factor_mesh, but with values scaled from 0 to 2pi
        """
        return self.factor_mesh_as_angles.reshape((self.n_data_points, self.n_factors))

    @property
    def flat_images(self) -> np.ndarray:
        """
        Returns:
            flat_images: full dataset in the shape (n_data_points, h, w, d)
        """
        return self.images.reshape((self.n_data_points, *self.images.shape[-3:]))

    @property
    def flat_labels(self):
        """
        Returns:
            flat_images: full dataset in the shape (n_data_points, h, w, d)
        """
        if self.labels is None:
            print("No class_labels available")
            return self.labels
        else:
            return self.labels.reshape(self.n_data_points)

    def setup_circles_dataset_labelled_pairs(self, n_labels: int) -> (np.ndarray, List[np.ndarray], np.ndarray):
        """
        Args:
            n_labels: Number of labelled pairs to generate
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
        assert 2 * n_labels <= self.n_data_points, \
            "for this procedure 2 * n_labels cannot exceed the number of data points"
        flat_factor_mesh = self.flat_factor_mesh
        # sample 2*n_labels indices, for the data points/pairs to be labelled
        indices = np.random.choice(self.n_data_points, size=2 * n_labels, replace=False)
        # split in two halves, for the first and second elements of the pairs
        ind1 = indices[:n_labels]
        ind2 = indices[n_labels:]

        x_l_transformations = []
        for factor_num in range(self.n_factors):
            if self.factors_shape[factor_num] != 1:
                differences = (flat_factor_mesh[ind2, factor_num] - flat_factor_mesh[ind1, factor_num]) % \
                              self.max_factor_values[factor_num]
                angles = np.expand_dims(2 * np.pi * differences / self.max_factor_values[factor_num], axis=1)
                identity_transformations = np.zeros_like(angles)
                x_l_transformations.append(np.stack([identity_transformations, angles], axis=1))

        # set up the set x_l of labelled data points, with shape (n_labels, 2, height, width, depth)
        images_flat = self.flat_images
        x1 = images_flat[ind1]  # shape (n_labels, height, width, depth)
        x2 = images_flat[ind2]  # shape (n_labels, height, width, depth)
        x_l = np.stack([x1, x2], axis=1)  # shape (n_labels, 2, height, width, depth)

        # select all remaining data points for the unlabelled set x_u,
        #   with shape (n_unlabelled, 1, height, width, depth)
        mask = np.ones(self.n_data_points, dtype=bool)
        mask[indices] = False
        x_u = images_flat[mask]
        x_u = np.expand_dims(x_u, axis=1)  # shape (n_data_points - 2*n_labels, 1, height, width, depth)
        x_l_transformations = tuple(x_l_transformations)
        return x_l, x_l_transformations, x_u

    def setup_cylinder_dataset_labelled_pairs(self, n_pairs: int, angle_factor: int = 1) -> (np.ndarray, List[np.ndarray], np.ndarray):
        """
        For a dataset with objects in the first dimension, creates a set of x_l labeled pairs with x_l_transformations
        and a set of x_u unlabeled data
        Args:
            n_pairs: number of pairs per object
            angle_factor: index of the factor that contains the rotation information in the factor_mesh

        Returns:
            x_l: Labeled pairs array with shape (n_labels, 2, height, width, depth)
            x_l_transformations: List of length n_factors, each element is an array of shape (n_labels, 2, 1)
                where [:, 0, :] represents the identity transformations,
                and [:, 1, :] represents the transformation from the first to the second element of a pair,
                given as an angle on the unit circle
            x_u: Unlabeled data points with shape (n_data_points - 2*n_labels, 1, height, width, depth)

        """
        # TODO: actually the arrays in x_l_transformations have shape (n_labels, 2) without the extra axis. Is that ok?
        x_l = []
        x_u = []
        x_l_transformations = []

        for num_object, object in enumerate(self.images):
            indexes = np.arange(len(object))
            np.random.shuffle(indexes)
            labeled_indexes = indexes[:2 * n_pairs]
            labels = self.factor_mesh[num_object, labeled_indexes, angle_factor]
            labels = labels.reshape((-1, 2))
            differences = (labels[:, 1] - labels[:, 0]) % self.max_factor_values[angle_factor]
            angles = np.expand_dims(2 * np.pi * differences / self.max_factor_values[angle_factor], axis=1)
            angles = angles.reshape((-1))
            identity_transformation = np.zeros_like(angles)
            x_l_transformations.append(np.stack([identity_transformation, angles], axis=1))
            unlabeled_indexes = indexes[2 * n_pairs:]
            x_l.append(object[labeled_indexes].reshape((-1, 2, *self.image_shape)))
            x_u.append(object[unlabeled_indexes].reshape((-1, 1,  *self.image_shape)))
        x_u = np.concatenate(x_u, axis=0)
        x_l = np.concatenate(x_l, axis=0)
        x_l_transformations = np.concatenate(x_l_transformations, axis=0)
        x_l_transformations = [np.zeros_like(x_l_transformations), x_l_transformations]
        return x_l, tuple(x_l_transformations), x_u

    def separate_by_labels(self):
        images_per_class = []
        labels_per_class = []
        mesh_per_class = []
        for num_label, label in enumerate(np.unique(self.flat_labels)):
            images_per_class.append(self.images[self.labels[:, 0] == label])
            labels_per_class.append(self.labels[self.labels[:, 0] == label])
            mesh_per_class.append(self.factor_mesh[self.labels[:, 0] == label])
            print(f"Num objects per class {num_label}:", len(images_per_class[-1]))
        return images_per_class, labels_per_class, mesh_per_class

    def random_paths_torus(self, n_paths, path_length, step_size=1):
        """
        Generate random paths with random starting points, adjacent path elements are exactly step_size steps away in
        exactly 1 factor, and have all other factors equal. Only works for datasets where all factors are meaningful as
        angles (i.e. torus datasets).

        Args:
            n_paths (int): number of paths to generate
            path_length (int): length of each path
            step_size (int): step size between two path instances. Should be relatively prime to the number of instances
                for each factor, to ensure that the entire dataset can be reached.
        Returns:
            images_paths: array with images from the random paths (n_paths, path_length, height, width, depth)
            transformations: List of length n_factors, each element is an array of shape (n_paths, path_length, 1)
                where [:, 0, :] represents the identity transformations,
                and [:, i, :] represents the transformation from the first to the i-th element of a path,
                given as an angle on the unit circle
        """
        # starting points, shape (n_paths, 1, n_factors), axis1 is for broadcasting
        start = [[np.random.randint(factor_size) for factor_size in self.factors_shape] for _ in range(n_paths)]
        start = np.expand_dims(start, axis=1)

        # random integers to choose which factor should make a step
        random_factors = np.random.randint(self.n_factors, size=(n_paths, path_length))

        # random +/-1 values to choose whether step goes forwards or backwards
        random_plusminus = np.random.randint(2, size=(n_paths, path_length))
        random_plusminus[random_plusminus == 0] = -1
        random_plusminus[:, 0] = 0  # start each path with no transformation
        random_plusminus = random_plusminus * step_size  # replace +/- 1 with +/- step_size
        random_plusminus = np.expand_dims(random_plusminus, axis=2)

        # the actual steps to take, "one-hot" encoded (i.e. one factor has +/-1, all others 0),
        # shape (n_paths, path_length, n_factors)
        random_moves = to_categorical(random_factors, num_classes=self.n_factors, dtype="int")
        random_moves = random_plusminus * random_moves

        # the actual paths (indices), computed from the starting points, wrapped around the max factor values,
        # shape (n_paths, path_length, n_factors)
        random_paths_relative = np.cumsum(random_moves, axis=1)
        random_paths = random_paths_relative + start
        random_paths = np.mod(random_paths, self.factors_shape)

        # list of n_factor arrays of shape (n_paths, path_length, 1) representing the transformations between the path
        #   elements, given as angles from 0 to 2pi
        angles_relative = np.mod(2 * np.pi * random_paths_relative / self.factors_shape, 2 * np.pi)
        transformations = tuple([angles_relative[:, :, i:i+1] for i in range(self.n_factors)])

        # images corresponding to the paths, shape (n_paths, path_length, height, width, depth)
        images_paths = np.empty((n_paths, path_length, *self.image_shape))
        for path in range(n_paths):
            for step in range(path_length):
                img = self.images
                for index in random_paths[path, step]:
                    img = img[index]
                images_paths[path, step] = img

        return images_paths, transformations

    def random_paths_cylinder(self, n_paths, path_length, step_size=1):
        """
        Generate paths for cylinder datasets with shape (n_objects, factor_size, h, w, d), AKA cylinder datasets.
        Makes paths with random starting points, a random direction (positive or negative), and equally spaced points
        from the factor dimension (axis=1) with an in-between distance of step_size.

        Args:
            n_paths (int): number of paths to generate PER OBJECT
            path_length (int): length of each path
            step_size (int): step size between two path instances. Should be relatively prime to the number of factor
                values (factor_size), to ensure that the entire dataset can be reached.
        Returns:
            images_paths: array with images from the random paths (n_paths * n_objects, path_length, h, w, d)
            transformations: List of length 2, each element is an array of shape (n_paths * n_objects, path_length, 1)
                The first array element contains all zeros, reflecting that the object in each path is the same
                For the second array:
                    [:, 0, :] represents the identity transformations,
                    and [:, i, :] represents the transformation from the first to the i-th element of a path,
                    given as an angle on the unit circle
        """
        n_objects = self.factors_shape[0]
        factor_size = self.factors_shape[1]

        # starting points, shape (n_objects, n_paths, 1), axis2 is for broadcasting
        start = np.random.randint(factor_size, size=(n_objects, n_paths))
        start = np.expand_dims(start, axis=2)

        # array of random directions +/-1, shape (n_objects, n_paths, 1), axis2 is for broadcasting
        direction = np.random.randint(2, size=(n_objects, n_paths))  # 0 or 1
        direction = 2 * direction - 1  # -1 or 1
        direction = np.expand_dims(direction, axis=2)

        # array of shape (1, 1, path_length) representing a forward path that starts at 0, axes0/1 are for broadcasting
        path_relative = np.arange(0, path_length * step_size, step_size)
        path_relative = np.expand_dims(np.expand_dims(path_relative, axis=0), axis=0)

        # relative paths (starting at 0) with randomised directions. Shape: (n_objects, n_paths, path_length)
        paths_relative = direction * path_relative

        # absolute paths with correct starting points, random directions, and modulo factor_size
        #   shape: (n_objects, n_paths, path_length)
        paths = paths_relative + start
        paths = np.mod(paths, factor_size)

        # list of n_factor arrays of shape (n_paths * n_objects, path_length) representing the transformations between
        #   the path elements, given as angles from 0 to 2pi
        angles_relative = np.mod(2 * np.pi * paths_relative / factor_size, 2 * np.pi)
        transformations = np.reshape(angles_relative, (n_objects * n_paths, path_length))
        transformations = tuple([np.zeros_like(transformations), transformations])

        # images corresponding to the paths, shape (n_paths * n_objects, path_length, height, width, depth)
        images_paths = np.empty((n_objects, n_paths, path_length, *self.image_shape))
        for object_id in range(n_objects):
            for path in range(n_paths):
                for step in range(path_length):
                    factor_id = paths[object_id, path, step]
                    images_paths[object_id, path, step] = self.images[object_id, factor_id]
        images_paths = np.reshape(images_paths, (n_objects * n_paths, path_length, *self.image_shape))

        return images_paths, transformations
