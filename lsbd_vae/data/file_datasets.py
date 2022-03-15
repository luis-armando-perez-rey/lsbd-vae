import tensorflow as tf
from typing import List, Tuple, Optional, Callable
import numpy as np
import os


def load_images_paths(flat_paths, image_shape):
    images = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for path in flat_paths:
        image = tf.image.decode_image(tf.io.read_file(path))[:, :, :image_shape[-1]]
        image = tf.cast(image, tf.float32)
        images = images.write(images.size(), image / 255.)
    images = images.stack()
    return images


class PathsTransformationsLabels:
    def __init__(self, paths: Optional[List], transformations: Optional[List], labels: Optional[List],
                 image_shape: Tuple[int, int, int]):

        self.image_shape = image_shape
        self.paths = paths
        self.transformations = transformations
        self.labels = labels

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value: Tuple[int, int, int]):
        self._image_shape = value

    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, values):
        self._paths = tf.concat(values, axis=0)

    @property
    def transformations(self):
        return self._transformations

    @transformations.setter
    def transformations(self, values):
        if values is None:
            self._transformations = None
        else:
            transfs = [np.array(transformation) for transformation in values]
            transfs = tf.concat(transfs, axis=1)
            self._transformations = tuple([transf for transf in transfs])

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        if values is None:
            self._labels = None
        else:
            self._labels = tf.concat(values, axis=0)

    def __get_image_load_function(self, batch_size: int, paths_shape: Tuple, get_labels: bool = False,
                                  get_transformations: bool = False,
                                  flatten: bool = False):
        """
        Create function to apply to tf.data.Dataset data
        :param batch_size: batch size output of the tf.data.Dataset
        :param get_labels: whether to output the labels
        :param flatten: whether the images should be output with shape (bath_size * length_random_walk, *image_shape)
        :return:
        """
        paths_per_group = paths_shape[1]
        un_flatten = not flatten

        def image_load(features):
            paths = features["paths"]
            flat_paths = tf.reshape(paths, (batch_size * paths_per_group,))

            # Get images
            imgs = load_images_paths(flat_paths, self.image_shape)
            if un_flatten:
                imgs = tf.reshape(imgs, (batch_size, paths_per_group, *self.image_shape))
            output = {"images": imgs}

            # Get labels
            if get_labels:
                labels = features["labels"]
                if flatten:
                    labels = tf.reshape(labels, (batch_size * paths_per_group,))
                output.update({"labels": labels})

            # Get transformations
            if get_transformations:
                transformations = features["transformations"]
                if flatten:
                    transformations = tuple(
                        [tf.reshape(transformation, (batch_size * paths_per_group,)) for transformation in
                         transformations])
                output.update({"transformations": transformations})
            return output, None  # output tuple since Tensorflow is accostumed to be fit with x,y data

        return image_load

    def get_tfdataset(self, batch_size: int, shuffle: int = 1000, flatten: bool = False) -> tf.data.Dataset:
        """
        Get tensorflow dataset from random walk
        :param flatten: whether to provide a tfdataset that outputs flattened data i.e. with shape
        (num_random_walks * random_walk_length, *image_shape)
        :param batch_size:
        :param shuffle: number of datapoints to be taken during shuffle if equal to 0 then no shuffle is added
        :return:
        """
        if len(self.paths) % batch_size != 0:
            print(
                f"WARNING: Number of path groups {len(self.paths)} is not a multiple of batch size {batch_size}."
                f"Some random walks will be dropped each iteration")

        data = {"paths": self.paths}
        if self.labels is not None:
            data.update({"labels": self.labels})
        if self.transformations is not None:
            data.update({"transformations": self.transformations})
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(
            self.__get_image_load_function(batch_size, paths_shape=self.paths.shape, get_labels=self.labels is not None,
                                           get_transformations=self.transformations is not None,
                                           flatten=flatten))
        # ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if shuffle != 0:
            ds = ds.shuffle(shuffle)
        return ds


class FactorCombinations(PathsTransformationsLabels):
    def __init__(self, source_path: str, factor_values_list: List,
                 image_shape: Tuple[int, int, int], factor_boolean: List[bool],
                 extension: str = ".png", labeling_function: Optional[Callable] = None):
        # Factor variables
        self.factor_values_list = factor_values_list
        self.num_factors = len(factor_values_list)
        self.factor_boolean = factor_boolean

        self.max_factor_indexes = np.array([len(factor) for factor in factor_values_list]).astype(np.int32)
        self.factor_indexes = [np.arange(max_factor_index) for max_factor_index in self.max_factor_indexes]
        self.factor_shapes = tuple([self.max_factor_indexes[num_factor] for num_factor in range(self.num_factors) if
                                    self.factor_boolean[num_factor]])
        self.identities_shape = tuple([self.max_factor_indexes[num_factor] for num_factor in range(self.num_factors) if
                                       not self.factor_boolean[num_factor]])

        self.total_identities = np.product(self.identities_shape)
        self.total_factor_combinations = np.product(self.factor_shapes)

        # Image variables
        self.image_shape = image_shape
        self.labeling_function = labeling_function

        # Path variables
        self.source_path = source_path
        self.extension = extension

        # Assertions
        assert np.ndim(factor_boolean) == 1, "Dimension of mask should be 1"
        assert len(factor_boolean) == len(
            factor_values_list), "Length of mask should be the same as the length of factor values"

        paths = self.__get_paths()
        labels = self.__get_labels(paths)
        transformations = None
        super(FactorCombinations, self).__init__([paths], transformations, labels, image_shape)

    def __gather_factor_string_meshgrid(self, factor_index_meshgrid):
        factor_string_meshgrid = []
        for num_factor in range(self.num_factors):
            # Gather factor values according to the corresponding factor states
            factor_values = tf.gather(self.factor_values_list[num_factor], factor_index_meshgrid[num_factor])
            factor_string_meshgrid.append(factor_values)
        return factor_string_meshgrid

    def __get_paths(self) -> tf.Tensor:
        """
        Takes factor_states and produces paths to the corresponding images
        :return:
        """
        # Make list of states with factor values per factor
        factor_index_meshgrid = tf.meshgrid(*self.factor_indexes, indexing="ij")
        factor_string_meshgrid = self.__gather_factor_string_meshgrid(factor_index_meshgrid)
        filenames = tf.strings.join(factor_string_meshgrid, separator="_")
        # Join states to image extension
        filenames = tf.strings.join([filenames, ".png"], separator="")
        # Create paths from source path
        paths = tf.strings.join([self.source_path, filenames], separator=os.sep)
        paths = tf.reshape(paths, (self.total_identities, *self.factor_shapes))
        return paths

    def __get_labels(self, paths) -> Optional[tf.Tensor]:
        """
        Gets the labels from the paths using the labeling function if defined.
        :return:
        """
        if self.labeling_function is None:
            labels = None
        else:
            labels = self.labeling_function(paths)
        return labels

    def __get_image_load_function_flat_paths(self, get_labels: bool = False):
        """
        Create function to apply to tf.data.Dataset data
        :param get_labels: whether to output the labels
        :return:
        """

        def image_load(features):
            flat_paths = features["paths"]
            images = load_images_paths(flat_paths, self.image_shape)
            labels = features["labels"]
            if get_labels:
                output = {"images": images, "labels": labels}
            else:
                output = {"images": images}
            return output, None  # output tuple since Tensorflow is accostumed to be fit with x,y data

        return image_load

    def get_tfdataset_flat(self, batch_size: int, shuffle: int = 0) -> tf.data.Dataset:
        """
        Get tensorflow dataset that produces ordered images which can be re-ordered into
        (n_identities, factor1, factor2, ... factorN, *image_shape) after complete data load. Shuffling makes the data
        lose the property of being re-ordered.
        :param batch_size: number of images to be output
        :param shuffle: number of data taken for each shuffle
        :return:
        """

        flat_paths = tf.reshape(self.paths, tf.reduce_prod(self.paths.shape))
        if flat_paths.shape[0] % batch_size != 0:
            print(
                f"WARNING: Number of random walks {self.total_identities} is not a multiple of batch size {batch_size}."
                f"Some random walks will be dropped each iteration")

        if self.labels is None:
            ds = tf.data.Dataset.from_tensor_slices({"paths": flat_paths})
        else:
            flat_labels = tf.reshape(self.labels, tf.reduce_prod(self.labels.shape))
            ds = tf.data.Dataset.from_tensor_slices({"paths": flat_paths, "labels": flat_labels})
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(self.__get_image_load_function_flat_paths(get_labels=self.labeling_function is not None))
        # ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if shuffle != 0:
            print(
                "Warning: Shuffle is not zero and data cannot be reorganized as (n_identities, factor1, factor2, ..."
                " factorN, *image_shape) after complete load. If this is intended then disregard this message")
            ds = ds.shuffle(shuffle)
        return ds


class RandomWalkFactor(PathsTransformationsLabels):
    """
    Class that can be used to create tf.data.Dataset that loads images within a source_path with naming structure
    factor1_factor2..._factornumfactor.extension by doing num_random_walks of length random_walk_length
    across the bool_change_factors. Class stores the paths corresponding to those images
    """

    def __init__(self, source_path: str, factor_values_list: List, num_random_walks: int,
                 step_sizes: np.array,
                 random_walk_length: int, bool_change_factors: List[bool], image_shape: Tuple[int, int, int],
                 extension: str = ".png", labeling_function: Optional[Callable] = None, seed: Optional[int] = None):
        """
        Initialize class
        :param source_path: path were images are stored
        :param factor_values_list: Possible factor values used in the names of the saved images
        :param num_random_walks: Number of random walks created
        :param step_sizes: Size of steps for each factor
        :param random_walk_length: Number of steps taken in each random walk
        :param bool_change_factors: List of booleans of the same length as factor values that indicates whether the
        factor changes during random walk
        :param image_shape: Shape of images to be loaded
        :param extension: Extensions of the images to be loaded
        """
        self.seed = seed
        tf.random.set_seed(self.seed)
        # Factor variables
        self.factor_values_list = factor_values_list
        self.num_factors = len(factor_values_list)

        # Random walk variables
        self.num_random_walks = num_random_walks
        self.max_factor_indexes = np.array([len(factor) for factor in factor_values_list]).astype(np.int32)
        self.random_walk_length = random_walk_length
        self.bool_change_factors = bool_change_factors
        self.num_changing_factors = sum(bool_change_factors)
        if len(step_sizes) == 1:
            self.step_sizes = np.concatenate([step_sizes] * self.num_factors, -1).astype(np.int32)
        else:
            self.step_sizes = step_sizes

        # Image variables
        self.image_shape = image_shape
        self.labeling_function = labeling_function

        # Path variables
        self.source_path = source_path
        self.extension = extension

        # Assertions
        assert np.ndim(bool_change_factors) == 1, "Dimension of mask should be 1"
        assert len(bool_change_factors) == len(
            factor_values_list), "Length of mask should be the same as the length of factor values"

        assert len(self.step_sizes) == self.num_factors

        paths, transformations, labels = self.__generate_image_paths_transformations()
        super(RandomWalkFactor, self).__init__([paths], [transformations], labels, image_shape)
        tf.random.set_seed(None)

    def get_random_initial(self) -> tf.Tensor:
        initial_factors = []
        for max_val in self.max_factor_indexes:
            random_initial = tf.random.uniform((self.num_random_walks, 1), maxval=max_val, dtype=tf.int32)
            initial_factors.append(random_initial)
        return tf.stack(initial_factors, axis=-1)

    def __get_random_step_sign(self) -> tf.Tensor:
        """
        Returns random +/- sign for factors that change
        :return:
        """
        sign = tf.random.uniform((self.num_random_walks, self.random_walk_length - 1), maxval=2, dtype=tf.int32)
        sign = tf.expand_dims(tf.where(sign == 0, x=tf.ones_like(sign), y=-tf.ones_like(sign)), axis=-1)
        return sign

    def __get_random_factors(self) -> tf.Tensor:
        """
        Samples uniform random values from factors that change
        :return:
        """
        return tf.random.uniform((self.num_random_walks, self.random_walk_length - 1), maxval=self.num_changing_factors,
                                 dtype=tf.int32)

    def __get_paths(self, factor_states) -> tf.Tensor:
        """
        Takes factor_states and produces paths to the corresponding images
        :param factor_states: shape (batch_size, n_steps, n_factors)
        :return:
        """
        # Make list of states with factor values per factor
        gathered_factors = []
        for num_factor in range(self.num_factors):
            # Gather factor values according to the corresponding factor states
            factor_values = tf.gather(self.factor_values_list[num_factor], factor_states[..., num_factor])
            gathered_factors.append(factor_values)
        # Join states in string
        filenames = tf.strings.join(gathered_factors, separator="_")
        # Join states to image extension
        filenames = tf.strings.join([filenames, self.extension], separator="")
        # Create paths from source path
        paths = tf.strings.join([self.source_path, filenames], separator=os.sep)
        return paths

    def __get_steps(self, random_factors, sign) -> tf.Tensor:
        """
        Creates the steps in the factor space by taking the sampled random factors and the random sign values
        :param random_factors: random factor identifiers representing the factor that changes in a given step
        (batch_size, n_steps)
        :param sign: +/- sign used for each transformation in the random walk
        :return:
        """
        # Transform random factors to steps
        steps = tf.keras.utils.to_categorical(random_factors, num_classes=self.num_changing_factors)
        #
        steps_list = []
        changing_factor_id = 0
        for num_factor, is_change in enumerate(self.bool_change_factors):
            if is_change:
                steps_list.append(steps[..., changing_factor_id])
                changing_factor_id += 1
            else:
                steps_list.append(tf.zeros((len(sign), self.random_walk_length - 1), dtype=tf.int32))
        steps = tf.stack(steps_list, axis=-1)
        steps = tf.expand_dims(tf.expand_dims(self.step_sizes, axis=0), axis=0) * sign * steps
        steps = tf.concat([tf.zeros((self.num_random_walks, 1, self.num_factors), dtype=tf.int32), steps], axis=1)
        return steps

    def __generate_image_paths_transformations(self) -> (tf.Tensor, Tuple, Optional[tf.Tensor]):
        """
        Creates the paths to the images, the transformations and provides the labels for the random walks
        :return:
        """
        # Starting point for each of the random walks
        starting_point = self.get_random_initial()  # (num_random_walks, num_steps, num_factors)
        # Get random factors modified in each step of the random walk
        random_factors = self.__get_random_factors()  # (batch_size, num_steps, 1)
        # Get random +/- sign for each step of the random walk
        sign = self.__get_random_step_sign()  # (batch_size, num_steps, 1)
        # Get the step values for each factor
        steps = self.__get_steps(random_factors, sign)  # (bath_size, num_steps, num_factors)
        # Accumulate the steps
        random_relative_steps = tf.cumsum(steps, axis=1)  # (batch_size, num_steps, num_factors)
        # Transform initial state
        final_states = random_relative_steps + starting_point  # (batch_size, num_steps, num_factors)
        # Wrap states according to max values
        final_states = tf.math.mod(final_states,
                                   tf.expand_dims(tf.expand_dims(self.max_factor_indexes, axis=0), axis=0))

        # Get relative transformations
        angles_relative = np.mod(2 * np.pi * tf.cast(random_relative_steps, tf.float32) / self.max_factor_indexes,
                                 2 * np.pi)
        transformations = tuple([angles_relative[:, :, i:i + 1] for i in range(self.num_factors)])

        # Get image paths
        paths = self.__get_paths(final_states)
        if self.labeling_function is None:
            labels = None
        else:
            labels = self.labeling_function(paths)
        return paths, transformations, labels


class RandomWalkIdentities(RandomWalkFactor):
    """
    Class that can be used to create tf.data.Dataset that loads images within a source_path with naming structure
    factor1_factor2..._factornumfactor.extension by doing random walks of length random_walk_lengthfor several
    fixed identities corresponding to all combinations of the bool_fixed_factors.
    Class stores the paths corresponding to those images.
    """

    def __init__(self, source_path: str, factor_values_list: List, num_random_walks_per_identity: int,
                 step_sizes: np.array,
                 random_walk_length: int, bool_change_factors: List[bool], image_shape: Tuple[int, int, int],
                 extension: str = ".png", labeling_function: Optional[Callable] = None, seed: Optional[int] = None):
        self.bool_fixed_factors = [not change_factor for change_factor in bool_change_factors]
        self.fixed_indexes_meshgrid = self.__get_fixed_indexes_mesh_grid(factor_values_list,
                                                                         num_random_walks_per_identity)
        total_random_walks = len(self.fixed_indexes_meshgrid[-1])
        print("Total random walks", total_random_walks)
        super(RandomWalkIdentities, self).__init__(source_path, factor_values_list,
                                                   total_random_walks, step_sizes,
                                                   random_walk_length, bool_change_factors, image_shape,
                                                   extension, labeling_function, seed)

    def __get_fixed_indexes_mesh_grid(self, factor_values_list, num_identity_repetitions):
        fixed_factors = []
        for num_factor, factor_values in enumerate(factor_values_list):
            if self.bool_fixed_factors[num_factor]:
                fixed_factors.append(np.arange(len(factor_values)))
        mesh_grids = np.array(np.meshgrid(*fixed_factors))
        fixed_factors_mesh_grid = []
        for grid in mesh_grids:
            flat_grid = grid.reshape(np.product(grid.shape))
            repeated_grid = np.tile(flat_grid, num_identity_repetitions)
            fixed_factors_mesh_grid.append(repeated_grid)
        return fixed_factors_mesh_grid

    def get_random_initial(self) -> tf.Tensor:
        initial_factors = []
        num_fixed = 0
        for num_factor, bool_change in enumerate(self.bool_change_factors):
            if bool_change:
                random_initial = tf.random.uniform((self.num_random_walks, 1),
                                                   maxval=self.max_factor_indexes[num_factor], dtype=tf.int32)
                initial_factors.append(random_initial)
            else:
                initial_factors.append(np.expand_dims(self.fixed_indexes_meshgrid[num_fixed], axis=-1))
                num_fixed += 1
        return tf.stack(initial_factors, axis=-1)


def join_file_datasets(ptl1: PathsTransformationsLabels,
                       ptl2: PathsTransformationsLabels) -> PathsTransformationsLabels:
    """
    Join two file datasets which are an instance of PathsTranfromationsLabels or any subclass.
    Produces a PathsTransformationsLabels that joins the paths, transformations and labels of the input classes.
    :param ptl1: Dataset 1
    :param ptl2: Dataset 2
    :return:
    """
    assert ptl1.image_shape == ptl2.image_shape, f"Image shapes must be the same"

    paths = [ptl1.paths, ptl2.paths]
    if ptl1.transformations is None or ptl2.transformations is None:
        transformations = None
    else:
        assert len(ptl1.transformations) == len(
            ptl2.transformations), f"Each dataset should have the same number of transformations"
        transformations = [ptl1.transformations, ptl2.transformations]
    if ptl1.labels is None or ptl2.labels is None:
        labels = None
    else:
        labels = [ptl1.labels, ptl2.labels]
    image_shape = ptl1.image_shape
    return PathsTransformationsLabels(paths, transformations, labels, image_shape)
