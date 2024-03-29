import numpy as np
import tensorflow as tf

from typing import List, Optional
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda
from tensorflow_probability import distributions as tfd

EPS = tf.keras.backend.epsilon()


class LatentSpace:
    """docstring for LatentSpace"""

    def __init__(self, kl_weight: float = 1.0, dist_weight: float = 1.0, aggregate_div_weight: float = 0.0,
                 name: str = ""):
        """

        Args:
            kl_weight: weight for kl loss term
            dist_weight:
            aggregate_div_weight:
            name:
        """
        self.kl_weight = kl_weight
        self.dist_weight = dist_weight
        self.aggregate_div_weight = aggregate_div_weight
        self.name = name  # useful to prevent using layer names multiple times, which gives an error

        # self.sampling, self.transform, self.inverse_transform, and self.average must be adjusted in each subclass
        self.sample_layer = Lambda(self.sampling, name=f"sample_{self.name}")

        self.transform_layer = Lambda(self.transform)  # layer with 2 inputs [z, transformations], outputs transformed z

        self.inverse_transform_layer = Lambda(self.inverse_transform)  # idem, but applies reverse of transformation

        self.avg_layer = Lambda(self.average)  # layer with input of shape (..., n_transformed_datapoints, z_dim),
        #           computes average over final axis and copies this such that the output shape equals the input shape

        # must be set within subclasses:
        self.params_layers = None  # list of layers, one for each parameter of the posterior
        self.transformation_shape = None  # shape tuple

        # TODO: Added locations and scale parameters to avoid indexing merge after tests and remove this comment
        self.loc_param_layer = None
        self.scale_param_layer = None

    @property
    def latent_dim(self):
        raise NotImplementedError()

    @staticmethod
    def projection(z_euclidean):
        """Projects z_euclidean onto the latent manifold. If the manifold is Euclidean,
         the projection is the identity"""
        raise z_euclidean

    def get_params(self, h_enc):
        return [param_layer(h_enc) for param_layer in self.params_layers]

    def sampling(self, enc_params):
        """Sampling function for the latent space. Takes encoder parameters and returns a sampled embedding"""
        raise NotImplementedError

    def kl_loss(self, enc_params):
        """KL Divergence from approximate posterior q(z|x) to prior p(x)"""
        raise NotImplementedError()

    def kl_loss_weighted(self, enc_params):
        return self.kl_weight * self.kl_loss(enc_params)

    def sample_from_prior(self, batch_shape):
        """sample random points from the prior, e.g. for generating data"""
        raise NotImplementedError()

    def distance(self, z1, z2):
        """"Computes 'distance' between z1 and z2, in a way that matches the latent space geometry
        z1 and z2 should have the same shape, distance is computed over the last axis/axes only
        (those representing the latent dimension). NOTE: actually represents squared distance"""
        raise NotImplementedError()

    def average(self, z):
        """"Computes 'average' of latent variables, over the last axis before those representing the latent dim
        (so typically over axis=-2), in a way that matches the latent space geometry
        z should have shape (..., n_vars_to_average, *latent_dims)"""
        raise NotImplementedError()

    def average_parameters(self, locations, scales):
        """
        Computes the parameters of the resulting distribution that "averages" the parameter estimates
        """
        raise NotImplementedError()

    def transform(self, z_and_transformations):
        """"Transforms z with given transformations"""
        raise NotImplementedError()

    def inverse_transform(self, z_and_transformations):
        """"Transforms z with inverse of given transformations"""
        raise NotImplementedError()

    def log_q_zkni_nj(self, enc_params_and_sample):
        """ Computes array of shape (M, M, K) for indices (i, j, k),
            where entry ijk represents log q( z_k(n_i) | n_j ),
                where z(n_i) is a (k-dimensional) sample from q(z|n_i),
                M = batch size,
                K = self.dim (number of dimensions),
            Needed to compute E_q(z) [log q(z)] for all latent spaces together
        """
        raise NotImplementedError()

    def crossentropy_posterior_prior(self, enc_params):
        """ Computes - E_q(z|x) [ log p(z) ] either exactly or with a single-sample estimation for E_q(z|x)
            output shape: (batch_size,)
        """
        raise NotImplementedError()


# LATENT SPACE SUBCLASSES
class GaussianLatentSpace(LatentSpace):
    """docstring for GaussianLatentSpace"""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        self.loc_param_layer = Dense(self.dim, name=f"guassian_mu_{self.name}")
        self.scale_param_layer = Dense(self.dim, name=f"gaussian_log_sigma_{self.name}",
                                       bias_initializer=tf.keras.initializers.zeros,
                                       kernel_initializer=tf.keras.initializers.zeros)

        self.params_layers = [self.loc_param_layer, self.scale_param_layer]
        self.transformation_shape = (1,)

    @property
    def latent_dim(self):
        return self.dim

    def sampling(self, enc_params):
        mu, log_sigma = enc_params
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=K.shape(mu))
        z_sample = mu + tf.math.exp(log_sigma) * epsilon
        return z_sample

    def kl_loss(self, enc_params):
        mu, log_sigma = enc_params
        kl_loss = - 0.5 * K.sum(1 + 2 * log_sigma - K.square(mu) - K.exp(2 * log_sigma), axis=-1)
        return kl_loss

    def sample_from_prior(self, batch_shape):
        return np.random.normal(size=batch_shape + (self.dim,), loc=0, scale=1)

    def average(self, z):
        n_transformations = int(z.shape[-2])  # z has shape (*batch_dims, n_transformations, latent_dim)
        z_sum = K.sum(z, axis=-2, keepdims=True)
        z_avg = z_sum / n_transformations
        z_avg = K.repeat_elements(z_avg, n_transformations, axis=-2)
        return z_avg

    def average_parameters(self, locations, log_scales):
        n_transformations = int(locations.shape[-2])
        loc_mean = tf.reduce_mean(locations, axis=-2, keepdims=True)
        loc_mean = tf.keras.backend.repeat_elements(loc_mean, n_transformations, axis=-2)
        log_scale_mean = tf.math.log(tf.reduce_mean(tf.math.exp(log_scales), axis=-2, keepdims=True))
        log_scale_mean = tf.keras.backend.repeat_elements(log_scale_mean, n_transformations, axis=-2)
        return loc_mean, log_scale_mean

    def log_q_zkni_nj(self, enc_params_and_sample):
        """Array of shape (M, M, K) for indices (i, j, k),
            where entry ijk represents log q( z_k(n_i) | n_j ),
                where z(n_i) is a (k-dimensional) sample from q(z|n_i).
                M = batch size
                K = self.dim (number of dimensions)
            Needed to compute E_q(z) [log q(z)] for all latent spaces
        """
        mu, log_sigma, sample = enc_params_and_sample
        q_z_given_x = tfd.Normal(loc=mu, scale=K.exp(log_sigma))
        # ideally you want to copy sample M times, resulting in shape (M, M, K)
        #   but at this point it's not possible to obtain the batch_size M (it's still None)
        # trick: change to shape (M, 1, K), and log_prob will act the same as if the shape were (M, M, K)
        #        because of broadcasting
        sample_repeated = K.repeat(sample, 1)  # resulting in shape (M, 1, K)
        return q_z_given_x.log_prob(sample_repeated)

    def inverse_transform(self, z_and_transformations):
        """For Euclidean latent space apply identity"""
        z, transformations = z_and_transformations
        return z

    def transform(self, z_and_transformations):
        """For Euclidean latent space apply identity"""
        z, transformations = z_and_transformations
        return z

    def distance(self, z1, z2):
        return tf.norm(z1 - z2, ord="euclidean", axis=-1)

    def crossentropy_posterior_prior(self, enc_params):
        """ Computes - E_q(z|x) [ log p(z) ] exactly as cross-entropy
            output shape: (batch_size,)
        """
        mu, log_sigma = enc_params
        p_z = tfd.Normal(loc=0, scale=1)  # prior distribution p(z)
        q_z_given_x = tfd.Normal(loc=mu, scale=K.exp(log_sigma))  # posterior distribution q(z|x)
        crossentropy_per_dim = q_z_given_x.cross_entropy(p_z)  # has shape (batch_size, self.dim)
        return K.sum(crossentropy_per_dim, axis=1)  # has shape (batch_size,)


# TODO: Fix deterministic models to make them run appropriately with this latent space
class DiracDeltaLatentSpace(LatentSpace):
    """Latent space used for Autoencoders with deterministic distribution of encodings"""

    def crossentropy_posterior_prior(self, enc_params):
        pass

    def log_q_zkni_nj(self, enc_params_and_sample):
        pass

    def sample_from_prior(self, batch_shape):
        pass

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        self.loc_param_layer = Dense(self.dim, name=f"mu_{self.name}")
        self.scale_param_layer = None
        self.params_layers = [self.loc_param_layer]
        self.transformation_shape = (1,)

    @property
    def latent_dim(self):
        return self.dim

    def sampling(self, enc_params):
        """
        Deterministic output, assume first argument is location
        """
        mu = enc_params[0]
        return mu

    def kl_loss(self, enc_params):
        kl_loss = 0.0
        return kl_loss

    def average(self, z):
        n_transformations = int(z.shape[-2])  # z has shape (*batch_dims, n_transformed_datapoints, latent_dim)
        z_sum = K.sum(z, axis=-2, keepdims=True)
        z_avg = z_sum / n_transformations
        z_avg = K.repeat_elements(z_avg, n_transformations, axis=-2)
        return z_avg

    def average_parameters(self, locations, scales):
        n_transformations = int(locations.shape[-2])
        loc_mean = tf.reduce_mean(locations, axis=-2, keepdims=True)
        loc_mean = tf.keras.backend.repeat_elements(loc_mean, n_transformations, axis=-2)
        return loc_mean, scales

    def inverse_transform(self, z_and_transformations):
        """For Euclidean latent space apply identity"""
        z, transformations = z_and_transformations
        return z

    def transform(self, z_and_transformations):
        """For Euclidean latent space apply identity"""
        z, transformations = z_and_transformations
        return z

    def distance(self, z1, z2):
        return tf.norm(z1 - z2, ord="euclidean", axis=-1)


class HyperSphericalLatentSpace(LatentSpace):

    def __init__(self, dim, steps=10, log_t_limit=(-10, -6), **kwargs):
        super(HyperSphericalLatentSpace, self).__init__(**kwargs)
        self.dim = dim
        self.steps = steps
        self.log_t_limit = log_t_limit

        if self.log_t_limit is None:
            print("Log t is not limited")
            log_t_layer = Dense(1, name=f"hyperspherical_log_t_{self.name}",
                                kernel_initializer=tf.keras.initializers.zeros)
        else:
            assert isinstance(self.log_t_limit, tuple), "Log t limits is not a tuple"
            assert len(self.log_t_limit) == 2, "Log t limits should be a 2-tuple"
            print(f"Min log t limit {self.log_t_limit[0]} Max log t limit {self.log_t_limit[1]}")

            def limit_log_t(x):
                half_time_interval_length = (self.log_t_limit[1] - self.log_t_limit[0]) / 2
                time_interval_center = (self.log_t_limit[1] + self.log_t_limit[0]) / 2
                return np.abs(half_time_interval_length) * tf.math.tanh(x) + time_interval_center

            log_t_layer = Dense(1, name=f"hyperspherical_log_t_{self.name}", activation=lambda x: limit_log_t(x),
                                bias_initializer=tf.initializers.zeros, kernel_initializer=tf.keras.initializers.zeros)

        # Parameters
        self.loc_param_layer = self.LocParamLayer(self.latent_dim, self.projection)
        self.scale_param_layer = log_t_layer
        self.params_layers = [self.loc_param_layer, self.scale_param_layer]

        if self.latent_dim == 2:  # circular latent space, rotations are a single number in [0,2pi)
            self.transformation_shape = (1,)
        elif self.latent_dim == 3:  # spherical latent space, rotations can be defined by a rotation axis and an angle
            self.transformation_shape = (4,)  # e.g. [1, 0, 0, pi] for a half rotation around the x-axis
        else:
            self.transformation_shape = None

    @property
    def latent_dim(self):
        return self.dim + 1

    @staticmethod
    def projection(z_euclidean):
        z_projected = K.l2_normalize(z_euclidean, axis=-1)
        return z_projected

    def get_params(self, h_enc):
        # must be overwritten from subclass to handle the projection
        # mu_z_euclidean = self.params_layers[0](h_enc)  # locations parameter before projection
        mu_z = self.params_layers[0](h_enc)  # locations parameter after projection
        log_t = self.params_layers[1](h_enc)  # scale parameter
        return [mu_z, log_t]

    class LocParamLayer(tf.keras.layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def __init__(self, latent_dim, projection, **kwargs):
            super().__init__(**kwargs)
            self.latent_dim = latent_dim
            self.mu_z_euclidean = Dense(latent_dim)
            self.mu_z_layer = Lambda(projection)

        def call(self, inputs, **kwargs):
            return self.mu_z_layer(self.mu_z_euclidean(inputs))

        def compute_output_shape(self, input_shape):
            return input_shape[0], self.latent_dim

    def sampling(self, enc_params):
        mu_z, log_t = enc_params
        z_sample = mu_z
        for _ in range(self.steps):
            eps = K.random_normal(shape=K.shape(mu_z))
            step = K.exp(0.5 * log_t) * eps / np.sqrt(self.steps)
            z_sample = self.projection(z_sample + step)
        return z_sample

    def kl_loss(self, enc_params):
        mu_z, log_t = enc_params
        scalar_curv = self.dim * (self.dim - 1)
        volume = self.sphere_volume()
        kl_loss = - self.dim * log_t / 2.0 \
                  - self.dim * np.log(2.0 * np.pi) / 2.0 \
                  - self.dim / 2.0 \
                  + np.log(volume) \
                  + scalar_curv * K.exp(log_t) / 4.0
        kl_loss = K.squeeze(kl_loss, axis=-1)  # remove final dimension (which should always have size 1)
        return kl_loss

    def sample_from_prior(self, batch_shape):
        # TODO: make rejection sampling to prevent problems with near-zero norms
        random_normal = np.random.normal(size=batch_shape + (self.latent_dim,), loc=0, scale=1)
        norm = np.linalg.norm(random_normal)
        return random_normal / norm

    def distance(self, z1, z2):
        # z1 and z2 are points on the unit dim-sphere so no need to normalise them first
        cos_similarity = K.sum(z1 * z2, axis=-1)
        cos_similarity = K.clip(cos_similarity, -1, 1)  # clip to fix numerical errors
        return 2.0 * (1 - cos_similarity)

    @staticmethod
    def distance_angular(z1, z2):
        # z1 and z2 are points on the unit dim-sphere so no need to normalise them first
        cos_similarity = K.sum(z1 * z2, axis=-1)
        cos_similarity = K.clip(cos_similarity, -1, 1)  # clip to fix numerical errors
        angular_distance = tf.acos(cos_similarity)
        return angular_distance

    def average(self, z):
        n_transformations = int(z.shape[-2])  # z has shape (*batch_dims, n_transformed_datapoints, latent_dim)
        z_sum = K.sum(z, axis=-2, keepdims=True)
        norm = tf.norm(z_sum, axis=-1, keepdims=True)
        z_avg = z_sum / norm
        z_avg = K.repeat_elements(z_avg, n_transformations, axis=-2)
        return z_avg

    def transform(self, z_and_transformations, inverse=False):
        z, transformations = z_and_transformations
        if self.latent_dim == 2:
            # z has shape (*batch_dims, 2) and transformations (*batch_dims, 1)
            # rotate each entry in z with the corresponding angle in transformations
            # rotation_matrices = tfg.geometry.transformation.rotation_matrix_2d.from_euler(transformations)
            rotation_matrices = make_rotation_matrix_2d(transformations)  # shape (*batch_dims, 2, 2)
            z = K.expand_dims(z, axis=-1)  # add dim of size 1 at the end, resulting shape (*batch_dims, 2, 1)
            # rotations matrices are orthogonal, so transposing will yield the inverse rotation
            z_rotated = tf.matmul(rotation_matrices, z, transpose_a=inverse)  # resulting shape (*batch_dims, 2, 1)
            z_rotated = tf.squeeze(z_rotated, axis=-1)  # resulting shape (*batch_dims, 2)
        elif self.latent_dim == 3:
            # z has shape (*batch_dims, 3) and transformations (*batch_dims, 4)
            # rotate each entry in z with the corresponding angle in transformations
            axis = transformations[..., :3]  # shape (*batch_dims, 3) TODO: check if this works correctly
            angle = transformations[..., 3:]  # shape (*batch_dims, 1) TODO: check if this works correctly
            # TODO: this may give the same error as rotation_matrix_2d did, in that case we need to copy the implem.
            # rotation_matrices = tfg.geometry.transformation.rotation_matrix_3d.from_axis_angle(axis, angle)
            rotation_matrices = from_axis_angle(axis, angle)
            z = K.expand_dims(z, axis=-1)  # add dim of size 1 at the end, resulting shape (*batch_dims, 3, 1)
            # rotations matrices are orthogonal, so transposing will yield the inverse rotation
            z_rotated = tf.matmul(rotation_matrices, z, transpose_a=inverse)  # resulting shape (*batch_dim, 3, 1)
            z_rotated = tf.squeeze(z_rotated, axis=-1)  # resulting shape (*batch_dims, 3)
        else:
            raise NotImplementedError()
        return z_rotated

    def inverse_transform(self, z_and_transformations):
        return self.transform(z_and_transformations, inverse=True)

    def sphere_volume(self):
        if self.latent_dim % 2 == 0:
            k = self.latent_dim / 2
            volume = self.latent_dim * np.pi ** k / np.math.factorial(k)
        else:
            k = (self.latent_dim - 1) / 2
            volume = self.latent_dim * 2 * np.math.factorial(k) * (4 * np.pi) ** k / np.math.factorial(2 * k + 1)
        return volume

    def crossentropy_posterior_prior(self, enc_params):
        pass

    def log_q_zkni_nj(self, enc_params_and_sample):
        pass

    def average_parameters(self, locations, scales):
        n_transformations = int(locations.shape[-2])
        loc_mean = tf.reduce_mean(locations, axis=-2, keepdims=True)
        loc_mean = self.projection(loc_mean)
        loc_mean = tf.keras.backend.repeat_elements(loc_mean, n_transformations, axis=-2)
        scale_mean = tf.reduce_mean(scales, axis=-2, keepdims=True)
        scale_mean = tf.keras.backend.repeat_elements(scale_mean, n_transformations, axis=-2)

        return loc_mean, scale_mean


def make_rotation_matrix_2d(angle):
    """"Replaces tfg.geometry.transformation.rotation_matrix_2d.from_euler since it gives an error"""
    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)
    output_shape = tf.concat((tf.shape(input=angle)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


# LATENT SPACE SUBCLASSES
class GaussianTorusLatentSpace(LatentSpace):
    """docstring for GaussianLatentSpace"""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        self.loc_param_layer = Dense(self.dim, name=f"guassian_mu_{self.name}")
        self.scale_param_layer = Dense(self.dim, name=f"gaussian_log_sigma_{self.name}",
                                       bias_initializer=tf.initializers.ones,
                                       kernel_initializer=tf.keras.initializers.zeros)
        self.params_layers = [self.loc_param_layer, self.scale_param_layer]

    @property
    def latent_dim(self):
        return self.dim

    def sampling(self, enc_params):
        mu, log_sigma = enc_params
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=K.shape(mu))
        z_sample = mu + K.exp(log_sigma) * epsilon
        return z_sample

    @staticmethod
    def projection(z_euclidean):
        z_projected = K.l2_normalize(z_euclidean, axis=-1)
        return z_projected

    def kl_loss(self, enc_params):
        mu, log_sigma = enc_params
        kl_loss = - 0.5 * K.sum(1 + 2 * log_sigma - K.square(mu) - K.exp(2 * log_sigma), axis=-1)
        return kl_loss

    def sample_from_prior(self, batch_shape):
        return np.random.normal(size=batch_shape + (self.dim,), loc=0, scale=1)

    def log_q_zkni_nj(self, enc_params_and_sample):
        """Array of shape (M, M, K) for indices (i, j, k),
            where entry ijk represents log q( z_k(n_i) | n_j ),
                where z(n_i) is a (k-dimensional) sample from q(z|n_i).
                M = batch size
                K = self.dim (number of dimensions)
            Needed to compute E_q(z) [log q(z)] for all latent spaces
        """
        mu, log_sigma, sample = enc_params_and_sample
        q_z_given_x = tfd.Normal(loc=mu, scale=K.exp(log_sigma))
        # ideally you want to copy sample M times, resulting in shape (M, M, K)
        #   but at this point it's not possible to obtain the batch_size M (it's still None)
        # trick: change to shape (M, 1, K), and log_prob will act the same as if the shape were (M, M, K)
        #        because of broadcasting
        sample_repeated = K.repeat(sample, 1)  # resulting in shape (M, 1, K)
        return q_z_given_x.log_prob(sample_repeated)

    def crossentropy_posterior_prior(self, enc_params):
        """ Computes - E_q(z|x) [ log p(z) ] exactly as cross-entropy
            output shape: (batch_size,)
        """
        mu, log_sigma = enc_params
        p_z = tfd.Normal(loc=0, scale=1)  # prior distribution p(z)
        q_z_given_x = tfd.Normal(loc=mu, scale=K.exp(log_sigma))  # posterior distribution q(z|x)
        crossentropy_per_dim = q_z_given_x.cross_entropy(p_z)  # has shape (batch_size, self.dim)
        return K.sum(crossentropy_per_dim, axis=1)  # has shape (batch_size,)

    def distance(self, z1, z2):
        pass

    def inverse_transform(self, z_and_transformations):
        pass

    def transform(self, z_and_transformations):
        pass

    def average_parameters(self, locations, scales):
        pass

    def average(self, z):
        pass


def assert_normalized(vector,
                      order='euclidean',
                      axis=-1,
                      eps=None,
                      name='assert_normalized'):
    """Checks whether vector/quaternion is normalized in its last dimension.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    vector: A tensor of shape `[A1, ..., M, ..., An]`, where the axis of M
      contains the vectors.
    order: Order of the norm passed to tf.norm.
    axis: The axis containing the vectors.
    eps: A `float` describing the tolerance used to determine if the norm is
      equal to `1.0`.
    name: A name for this op. Defaults to 'assert_normalized'.
  Raises:
    InvalidArgumentError: If the norm of `vector` is not `1.0`.
  Returns:
    The input vector, with dependence on the assertion operator in the graph.
  """

    with tf.name_scope(name):
        vector = tf.convert_to_tensor(value=vector)
        if eps is None:
            eps = select_eps_for_division(vector.dtype)
        eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

        norm = tf.norm(tensor=vector, ord=order, axis=axis)
        one = tf.constant(1.0, dtype=norm.dtype)
        with tf.control_dependencies(
                [tf.debugging.assert_near(norm, one, atol=eps)]):
            return tf.identity(vector)


def from_axis_angle(axis, angle, name="rotation_matrix_3d_from_axis_angle"):
    """Convert an axis-angle representation to a rotation matrix.
  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
  Args:
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a normalized axis.
    name: A name for this op that defaults to
      "rotation_matrix_3d_from_axis_angle".
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represents a 3d rotation matrix.
  Raises:
    ValueError: If the shape of `axis` or `angle` is not supported.
  """
    with tf.name_scope(name):
        axis = tf.convert_to_tensor(value=axis)
        angle = tf.convert_to_tensor(value=angle)
        axis = assert_normalized(axis)

        sin_axis = tf.sin(angle) * axis
        cos_angle = tf.cos(angle)
        cos1_axis = (1.0 - cos_angle) * axis
        _, axis_y, axis_z = tf.unstack(axis, axis=-1)
        cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
        sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
        tmp = cos1_axis_x * axis_y
        m01 = tmp - sin_axis_z
        m10 = tmp + sin_axis_z
        tmp = cos1_axis_x * axis_z
        m02 = tmp + sin_axis_y
        m20 = tmp - sin_axis_y
        tmp = cos1_axis_y * axis_z
        m12 = tmp - sin_axis_x
        m21 = tmp + sin_axis_x
        diag = cos1_axis * axis + cos_angle
        diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
        matrix = tf.stack((diag_x, m01, m02,
                           m10, diag_y, m12,
                           m20, m21, diag_z),
                          axis=-1)  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)


def select_eps_for_division(dtype):
    """Selects default values for epsilon to make divisions safe based on dtype.
  This function returns an epsilon slightly greater than the smallest positive
  floating number that is representable for the given dtype. This is mainly used
  to prevent division by zero, which produces Inf values. However, if the
  nominator is orders of magnitude greater than `1.0`, eps should also be
  increased accordingly. Only floating types are supported.
  Args:
    dtype: The `tf.DType` of the tensor to which eps will be added.
  Raises:
    ValueError: If `dtype` is not a floating type.
  Returns:
    A `float` to be used to make operations safe.
  """
    return 10.0 * np.finfo(dtype.as_numpy_dtype).tiny
