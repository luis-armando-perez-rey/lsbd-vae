import tensorflow as tf
from typing import List, Tuple, Optional
import numpy as np
import os


def get_random_initial(num_paths: int, max_values) -> tf.Tensor:
    return tf.stack([tf.random.uniform((num_paths, 1), maxval=maxval, dtype=tf.int32) for maxval in max_values],
                    axis=-1)


class RandomWalkFactor:
    def __init__(self, source_path: str, factor_values: List, num_random_walks: int,
                 step_sizes: np.array,
                 random_walk_length: int, bool_change_factors: List[bool], image_shape: Tuple[int, int, int],
                 extension: str = ".png", labeling_function: Optional = None, seed: Optional = None):
        """
        Abstract class that can be used to create tf.data.Dataset that loads images with naming structure
        factor1_factor2..._factornumfactor.png
        :param source_path: path were images are stored
        :param factor_values: Possible factor values used in the names of the saved images
        :param num_random_walks: Number of random walks created
        :param step_sizes: Size of steps for each factor
        :param random_walk_length: Number of steps taken in each random walk
        :param bool_change_factors: List of booleans of the same length as factor values that indicates whether the
        factor changes during random walk
        :param image_shape: Shape of images to be loaded
        :param extension: Extensions of the images to be loaded
        """
        tf.random.set_seed(seed)
        # Factor variables
        self.factor_values = factor_values
        self.num_factors = len(factor_values)

        # Random walk variables
        self.num_random_walks = num_random_walks
        self.max_factor_indexes = np.array([len(factor) for factor in factor_values]).astype(np.int32)
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
            factor_values), "Length of mask should be the same as the length of factor values"

        assert len(self.step_sizes) == self.num_factors

        self.paths, self.transformations, self.labels = self._generate_image_paths_transformations()
        tf.random.set_seed(None)

    def _get_random_step_sign(self):
        """
        Returns random +/- sign for factors that change
        :return:
        """
        sign = tf.random.uniform((self.num_random_walks, self.random_walk_length - 1), maxval=2, dtype=tf.int32)
        sign = tf.expand_dims(tf.where(sign == 0, x=tf.ones_like(sign), y=-tf.ones_like(sign)), axis=-1)
        return sign

    def _get_random_factors(self):
        """
        Samples uniform random values from factors that change
        :return:
        """
        return tf.random.uniform((self.num_random_walks, self.random_walk_length - 1), maxval=self.num_changing_factors,
                                 dtype=tf.int32)

    def _get_paths(self, factor_states):
        """
        Takes factor_states and produces paths to the corresponding images
        :param factor_states: shape (batch_size, n_steps, n_factors)
        :return:
        """
        # Make list of states with factor values per factor
        gathered_factors = []
        for num_factor in range(self.num_factors):
            # Gather factor values according to the corresponding factor states
            factor_values = tf.gather(self.factor_values[num_factor], factor_states[..., num_factor])
            gathered_factors.append(factor_values)
        # Join states in string
        filenames = tf.strings.join(gathered_factors, separator="_")
        # Join states to image extension
        filenames = tf.strings.join([filenames, self.extension], separator="")
        # Create paths from source path
        paths = tf.strings.join([self.source_path, filenames], separator=os.sep)
        return paths

    def _get_steps(self, random_factors, sign):
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
                steps_list.append(tf.zeros((self.num_random_walks, self.random_walk_length - 1), dtype=tf.int32))
        steps = tf.stack(steps_list, axis=-1)
        steps = tf.expand_dims(tf.expand_dims(self.step_sizes, axis=0), axis=0) * sign * steps
        steps = tf.concat([tf.zeros((self.num_random_walks, 1, self.num_factors), dtype=tf.int32), steps], axis=1)
        return steps

    def _generate_image_paths_transformations(self):
        # Starting point for each of the random walks
        starting_point = get_random_initial(self.num_random_walks,
                                            self.max_factor_indexes)  # (batch_size, num_steps, num_factors)
        # Get random factors modified in each step of the random walk
        random_factors = self._get_random_factors()  # (batch_size, num_steps, 1)
        # Get random +/- sign for each step of the random walk
        sign = self._get_random_step_sign()  # (batch_size, num_steps, 1)
        # Get the step values for each factor
        steps = self._get_steps(random_factors, sign)  # (bath_size, num_steps, num_factors)
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
        paths = self._get_paths(final_states)
        if self.labeling_function is None:
            labels = None
        else:
            labels = self.labeling_function(paths)
        return paths, transformations, labels

    def _get_image_load_function(self, batch_size: int, get_labels: bool = False, flatten: bool = False):
        random_walk_length = self.random_walk_length

        def image_load(features):
            paths = features["paths"]
            transformations = features["transformations"]

            flat_paths = tf.reshape(paths, (batch_size * random_walk_length,))
            imgs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for path in flat_paths:
                image = tf.image.decode_image(tf.io.read_file(path))[:, :, :self.image_shape[-1]]
                image = tf.cast(image, tf.float32)
                imgs = imgs.write(imgs.size(), image / 255.)
            imgs = imgs.stack()
            if get_labels:
                labels = features["labels"]
                if flatten:
                    imgs = tf.reshape(imgs, (batch_size * random_walk_length, *self.image_shape))
                    labels = tf.reshape(labels, (batch_size * random_walk_length,))
                else:
                    imgs = tf.reshape(imgs, (batch_size, random_walk_length, *self.image_shape))
                output = {"images": imgs, "transformations": transformations, "labels": labels}
            else:
                if flatten:
                    imgs = tf.reshape(imgs, (batch_size * random_walk_length, *self.image_shape))
                else:
                    imgs = tf.reshape(imgs, (batch_size, random_walk_length, *self.image_shape))
                output = {"images": imgs, "transformations": transformations}
            return output

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
        if self.num_random_walks % batch_size != 0:
            print(
                f"WARNING: Number of random walks {self.num_random_walks} is not a multiple of batch size {batch_size}."
                f"Some random walks will be dropped each iteration")

        if self.labels is None:
            ds = tf.data.Dataset.from_tensor_slices({"paths": self.paths, "transformations": self.transformations})
        else:
            ds = tf.data.Dataset.from_tensor_slices(
                {"paths": self.paths, "transformations": self.transformations, "labels": self.labels})
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(
            self._get_image_load_function(batch_size, get_labels=self.labeling_function is not None, flatten=flatten))
        # ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if shuffle != 0:
            ds = ds.shuffle(shuffle)
        return ds
