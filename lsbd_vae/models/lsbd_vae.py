import tensorflow as tf
import numpy as np
from typing import List, Tuple, Optional
from lsbd_vae.models.reconstruction_losses import gaussian_loss
from lsbd_vae.models.latentspace import LatentSpace


class BaseLSBDVAE(tf.keras.Model):
    """
    LSBDVAE Variational Autoencoder which creates equivariant transformations. The model can be trained in
    semi-supervised fashion.
    """

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def __init__(self, encoder_backbones: List[tf.keras.models.Model], decoder_backbone: tf.keras.models.Model,
                 latent_spaces: List[LatentSpace], input_shape: Tuple[int, int, int],
                 reconstruction_loss=gaussian_loss, stop_gradient: bool = False, **kwargs):

        super(BaseLSBDVAE, self).__init__(**kwargs)

        # Parameters
        self.reconstruction_loss = reconstruction_loss(n_data_dims=len(input_shape))
        self.stop_gradient = stop_gradient
        self.input_shape_ = input_shape

        # Latent spaces
        self.latent_spaces = latent_spaces

        # Backbones
        self.encoder_backbones = encoder_backbones
        self.decoder = decoder_backbone
        assert self.decoder.inputs[0].shape[-1] == self.latent_dim,\
            "The decoder backbone should receive tensors of shape (batch_size, latent_dim)"

        # Set networks
        self.lst_encoder_loc, self.lst_encoder_scale = self.set_lst_parameter_encoders()
        self.encoder_flat = self.set_encoder_flat()
        self.decoder_from_list = self.set_decoder_from_list()
        self.loss_model = self.set_loss_model()

        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.equivariance_tracker = tf.keras.metrics.Mean(name="equivariance_loss")

        # Assertions
        assert len(self.encoder_backbones) == len(
            self.latent_spaces), "list of encoders and latent_spaces must have equal length"
        for num_encoder, encoder_backbone in enumerate(encoder_backbones):
            assert tf.keras.backend.int_shape(encoder_backbone.input)[
                   1:] == self.input_shape_, f"encoder {num_encoder} has different input shape"
            assert len(encoder_backbone.output.shape[1:]) == 1, "encoder {} output must be flattened," \
                                                                "i.e. have shape (batch_size, dim)"
        assert len(self.decoder.input.shape[
                   1:]) == 1, "decoder input must be flattened, i.e. have shape (batch_size, dim)"

    @property
    def latent_dim(self) -> int:
        return sum([latent_space.latent_dim for latent_space in self.latent_spaces])

    @property
    def n_latent_spaces(self) -> int:
        return len(self.latent_spaces)

    @property
    def encoder_backbones(self) -> List[tf.keras.models.Model]:
        return self.__encoder_backbones

    @encoder_backbones.setter
    def encoder_backbones(self, encoder_backbones) -> None:
        if len(encoder_backbones) == 1:
            self.__encoder_backbones = encoder_backbones * len(self.latent_spaces)
        else:
            assert len(self.latent_spaces) == len(encoder_backbones), (f"Number of encoder backbones "
                                                                       f"{len(encoder_backbones)} is not the same as"
                                                                       f" the number of latent spaces"
                                                                       f" {len(self.latent_spaces)}")
            self.__encoder_backbones = encoder_backbones

    def set_lst_parameter_encoders(self) -> Tuple[List[tf.keras.Model], List[tf.keras.Model]]:
        lst_encoder_loc = []
        lst_encoder_scale = []
        # Define encoder backbone
        input_encoder = tf.keras.layers.Input(self.input_shape_)

        for encoder_backbone, latent_space in zip(self.encoder_backbones, self.latent_spaces):
            h_enc = encoder_backbone(input_encoder)
            lst_encoder_loc.append(tf.keras.Model(input_encoder, latent_space.loc_param_layer(h_enc)))
            lst_encoder_scale.append(tf.keras.Model(input_encoder, latent_space.scale_param_layer(h_enc)))

        return lst_encoder_loc, lst_encoder_scale

    def set_encoder(self, input_layer: tf.keras.layers.Layer):
        """
        Return encoder that processes input with shape (batch_size, num_transformations, *input_shape)
        :param input_layer: keras input layer with shape (num_transformations, *input_shape)
        """
        # Pass each image through the encoder, input_layer should have shape (num_transformations, *input_shape)
        lst_sample = []  # List of samples per latent space
        lst_loc = []  # List of loc parameter per latent space
        lst_scale = []  # List of scale parameter per latent space
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            # Estimate parameter tensors with shape (batch_size, n_transforms, param_shape)
            loc_param_estimate = tf.keras.layers.TimeDistributed(self.lst_encoder_loc[num_latent_space])(
                input_layer)
            lst_loc.append(loc_param_estimate)  # location parameter
            scale_param_estimate = tf.keras.layers.TimeDistributed(self.lst_encoder_scale[num_latent_space])(
                input_layer)
            lst_scale.append(scale_param_estimate)  # scale parameter
            # Sample
            lst_sample.append(latent_space.sampling([lst_loc[-1], lst_scale[-1]]))
        # Create encoder
        encoder = tf.keras.models.Model(input_layer, [lst_loc, lst_scale, lst_sample])
        return encoder

    def set_encoder_flat(self) -> tf.keras.models.Model:
        """
        Set the encoder model that can receive images with shape (n_images, *image_shape)
        """
        # Pass each image through the encoder, input_layer should have shape (num_transformations, *input_shape)
        input_layer = tf.keras.layers.Input(self.input_shape_)
        lst_sample = []  # List of samples per latent space
        lst_loc = []  # List of loc parameter per latent space
        lst_scale = []  # List of scale parameter per latent space
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            # Estimate parameter tensors with shape (batch_size, n_transforms, param_shape)
            loc_param_estimate = self.lst_encoder_loc[num_latent_space](input_layer)
            lst_loc.append(loc_param_estimate)  # location parameter
            scale_param_estimate = self.lst_encoder_scale[num_latent_space](input_layer)
            lst_scale.append(scale_param_estimate)  # scale parameter
            # Sample
            lst_sample.append(latent_space.sampling([lst_loc[-1], lst_scale[-1]]))
        # Create encoder
        encoder = tf.keras.models.Model(input_layer, [lst_loc, lst_scale, lst_sample])
        return encoder

    def set_encoder_single_view(self) -> tf.keras.models.Model:
        mult_input_layer = tf.keras.layers.Input((1, *self.input_shape_))
        return self.set_encoder(mult_input_layer)

    def set_encoder_multiview(self) -> tf.keras.models.Model:
        multi_input_layer = tf.keras.layers.Input((self.n_transforms, *self.input_shape_))
        return self.set_encoder(multi_input_layer)

    # @tf.function
    def kl_loss_function(self, loc_parameter_estimates, scale_parameter_estimates, use_kl_weight=True):
        kl_losses = []
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            if use_kl_weight:
                kl_func = latent_space.kl_loss_weighted
            else:
                kl_func = latent_space.kl_loss
            kl_loss_ = kl_func([loc_parameter_estimates[num_latent_space], scale_parameter_estimates[num_latent_space]])
            kl_losses.append(kl_loss_)
        kl_loss = tf.add_n(kl_losses)  # shape (*batch_dims)

        return kl_loss

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        list_metrics = [
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.equivariance_tracker,
        ]
        return list_metrics

    def calculate_loss_and_grads(self, reconstruction_loss, kl_loss, equivariance_loss, tape) -> None:
        total_loss = reconstruction_loss + kl_loss + equivariance_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

    def set_decoder_from_list(self):
        """
        Decoder model that can be applied directly on a list of encodings, so you don't need to concatenate encodings.
        This can help prevent Out-Of-Memory issues for larger datasets.
        """
        encodings_list_input = []
        for ls in self.latent_spaces:
            encodings_list_input.append(tf.keras.layers.Input((ls.latent_dim,)))
        dec_in = tf.keras.layers.Concatenate(-1)(encodings_list_input)
        dec_out = self.decoder(dec_in)
        return tf.keras.Model(encodings_list_input, dec_out)

    # region API Functions
    def encode_images(self, input_images) -> List:
        """
        Takes array of images (n_images, *image_shape) and encodes them into the location parameter of each encoder
        Args:
            input_images: array of shape (n_images, *image_shape)

        Returns:
            List of length n_latent_spaces; of encodings with shape (n_images, ls_latent_dim)
        """
        lst_loc, lst_scale, lst_sample = self.encoder_flat.predict(input_images)
        return lst_loc

    def encode_images_scale(self, input_images) -> List:
        """
        Takes array of images (n_images, *image_shape) and encodes them into the location parameter of each encoder
        Args:
            input_images: array of shape (n_images, *image_shape)

        Returns:
            List of length n_latent_spaces; of scale parameters with shape (n_images, scale_dim)
        """
        lst_loc, lst_scale, lst_sample = self.encoder_flat.predict(input_images)
        return lst_scale

    def encode_images_loc_scale(self, input_images) -> List:
        """
        Takes array of images (n_images, *image_shape) and encodes them into the location parameter of each encoder
        Args:
            input_images: array of shape (n_images, *image_shape)

        Returns:
            Two lists of length n_latent_spaces each; of encodings and scale parameters
        """
        lst_loc, lst_scale, lst_sample = self.encoder_flat.predict(input_images)
        return lst_loc, lst_scale

    def decode_latents(self, encodings_list: List[np.array]) -> np.array:
        """
        Takes list of latent encodings arrays and decodes them into images
        Args:
            encodings_list: list of arrays of shape (n_encodings, latent_dim_l), one for each latent space

        Returns:
            images array of shape (n_encodings, *image_shape)
        """
        assert len(encodings_list) == len(self.latent_spaces)
        return self.decoder_from_list.predict(encodings_list)

    def reconstruct_images(self, input_images: np.array, return_latents: bool) -> (np.array, Optional[List[np.array]]):
        """
        Encode images and decode them again into reconstructions. Optionally include the encodings.
        Args:
            input_images: array of shape (n_images, *image_shape)
            return_latents: if True, return latent encodings as well, otherwise only return reconstructions

        Returns:
            array of reconstructed images, shape (n_images, *image_shape)
            if return_latents:
                list of n_latent_spaces arrays of shape (n_images, latent_dim_l) of encodings per latent space
        """
        encodings_list = self.encode_images(input_images)
        reconstructions = self.decode_latents(encodings_list)
        if return_latents:
            return reconstructions, encodings_list
        else:
            return reconstructions

    def set_loss_model(self) -> tf.keras.Model:
        """
        Set model that can receive images with shape (n_images, *image_shape) and returns loss components
            [reconstruction_loss, kl_loss, elbo]. Can be used to obtain individual loss values for each input.
        """
        input_layer = tf.keras.layers.Input(self.input_shape_)
        lst_loc, lst_scale, lst_sample = self.encoder_flat(input_layer)
        reconstructions = self.decoder_from_list(lst_loc)
        reconstruction_losses = self.reconstruction_loss(input_layer, reconstructions)
        kl_losses = self.kl_loss_function(lst_loc, lst_scale, use_kl_weight=False)
        elbos = - reconstruction_losses - kl_losses
        return tf.keras.Model(input_layer, [reconstruction_losses, kl_losses, elbos])

    def compute_losses_and_elbos(self, input_images: np.array):
        return self.loss_model.predict(input_images)
    # endregion


class UnsupervisedLSBDVAE(BaseLSBDVAE):

    def __init__(self, encoder_backbones: List[tf.keras.models.Model], decoder_backbone: tf.keras.models.Model,
                 latent_spaces: List[LatentSpace], input_shape: Tuple[int, int, int],
                 reconstruction_loss=gaussian_loss, stop_gradient: bool = False, **kwargs):
        self.n_transforms = 1
        super(UnsupervisedLSBDVAE, self).__init__(encoder_backbones, decoder_backbone,
                                                  latent_spaces, input_shape,
                                                  reconstruction_loss, stop_gradient, **kwargs)
        self.encoder_single_view = self.set_encoder_single_view()
        self.decoder_unlabeled = self.set_decoder_unlabeled()

    def set_decoder_unlabeled(self) -> tf.keras.models.Model:
        # Pass multiple codes to decoder
        mult_input_layer = tf.keras.layers.Input((1, self.latent_dim))
        x = tf.keras.layers.TimeDistributed(self.decoder)(mult_input_layer)
        return tf.keras.Model(mult_input_layer, x)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # If one input is provided the training is unsupervised
            image_input = data[0]["images"]

            # Estimate encoder parameters and sample
            loc_parameter_estimates, scale_parameter_estimates, samples = self.encoder_single_view(image_input)
            z = tf.keras.layers.Concatenate(-1)(samples)
            # Reconstruction
            reconstruction = self.decoder_unlabeled(z)

            # Calculate reconstruction loss (SAME)
            reconstruction_loss = tf.reduce_mean(self.reconstruction_loss(image_input, reconstruction))

            # Calculate KL and equivariance loss (SAME)
            kl_loss = tf.reduce_mean(self.kl_loss_function(loc_parameter_estimates, scale_parameter_estimates))

            equivariance_loss = tf.zeros_like(kl_loss)

            # Total loss (SAME)
            self.calculate_loss_and_grads(reconstruction_loss, kl_loss, equivariance_loss, tape=tape)

            output_dictionary = dict(loss_s=self.total_loss_tracker.result(),
                                     reconstruction_loss_u=self.reconstruction_loss_tracker.result(),
                                     kl_loss_u=self.kl_loss_tracker.result())
        return output_dictionary

    def call(self, inputs, training=None, mask=None):
        return self.encoder_single_view(inputs)


class SupervisedLSBDVAE(BaseLSBDVAE):

    def __init__(self, encoder_backbones: List[tf.keras.models.Model], decoder_backbone: tf.keras.models.Model,
                 latent_spaces: List[LatentSpace], n_transforms: int, input_shape: Tuple[int, int, int],
                 reconstruction_loss=gaussian_loss, stop_gradient: bool = False, anchor_locations: bool = False,
                 **kwargs):
        super(SupervisedLSBDVAE, self).__init__(encoder_backbones, decoder_backbone,
                                                latent_spaces, input_shape,
                                                reconstruction_loss, stop_gradient, **kwargs)
        self.n_transforms = n_transforms
        self.anchor_locations = anchor_locations
        self.encoder_multiview = self.set_encoder_multiview()
        self.encoder_transformed = self.set_encoder_transformed()
        self.decoder_labeled = self.set_decoder_labeled()

    def set_encoder_multiview(self) -> tf.keras.models.Model:
        multi_input_layer = tf.keras.layers.Input((self.n_transforms, *self.input_shape_))
        return self.set_encoder(multi_input_layer)

    def set_encoder_transformed(self) -> tf.keras.models.Model:
        """
        Creates the encoder that incorporates inverse transformation for samples and averaging. Used only for data
        supervised with transformations. Transformation shape depends on the latent space.
        """
        # Define the input
        multi_input_layer = tf.keras.layers.Input((self.n_transforms, *self.input_shape_))

        lst_transformations = []  # List of transformations
        lst_sample_avg = []  # List of averaged latent after inverse transform average and transform
        lst_loc_avg = []  # List of averaged location after inverse transform average and transform
        lst_loc, lst_scale, lst_sample = self.encoder_multiview(multi_input_layer)
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            # Transformations
            assert latent_space.transformation_shape is not None, "Latent spaces with no transformation are not " \
                                                                  "supported "
            transformations = tf.keras.layers.Input(shape=(self.n_transforms,) + latent_space.transformation_shape,
                                                    name="t" + str(num_latent_space))
            lst_transformations.append(transformations)
            # Apply to sample inverse transform. If anchor_locations is selected then average inverse locations

            # Get loc latent average
            z_loc_anchored = latent_space.inverse_transform_layer(
                [lst_loc[num_latent_space], transformations])
            z_loc_anchored_avg = latent_space.avg_layer(z_loc_anchored)
            z_loc_avg = latent_space.transform_layer([z_loc_anchored_avg, transformations])
            lst_loc_avg.append(z_loc_avg)

            # Get sample average
            z_sample_anchored = latent_space.inverse_transform_layer(
                [lst_sample[num_latent_space], transformations])
            z_sample_anchored_avg = latent_space.avg_layer(z_sample_anchored)
            z_sample_avg = latent_space.transform_layer([z_sample_anchored_avg, transformations])
            if self.stop_gradient:
                z_sample_avg = tf.keras.layers.Lambda(lambda z: tf.keras.backend.stop_gradient(z))(z_sample_avg)
            lst_sample_avg.append(z_sample_avg)
        # Create encoder from z and y
        encoder = tf.keras.models.Model([multi_input_layer, *lst_transformations],
                                        [lst_loc, lst_scale, lst_loc_avg, lst_sample_avg, lst_sample])
        return encoder

    def set_decoder_labeled(self) -> tf.keras.models.Model:
        # Pass multiple codes to decoder
        mult_input_layer = tf.keras.layers.Input((self.n_transforms, self.latent_dim))
        x = tf.keras.layers.TimeDistributed(self.decoder)(mult_input_layer)
        return tf.keras.Model(mult_input_layer, x)

    @tf.function
    def equivariance_loss_function(self, z_non_avg, z, use_dist_weight=True):
        equivariance_losses = []
        for num_latent_space, latent_space in enumerate(self.latent_spaces):
            equivariance_loss_ = tf.reduce_sum(latent_space.distance(z_non_avg, z), axis=1)  # shape (*batch_dims)
            if use_dist_weight:
                equivariance_loss_ *= latent_space.dist_weight
            equivariance_losses.append(equivariance_loss_)
        equivariance_loss = tf.add_n(equivariance_losses)  # shape (*batch_dims)
        return equivariance_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            data = data[0]
            image_input = data["images"]
            transformations = data["transformations"]
            # Estimate encoder parameters and sample
            loc_parameter_estimates, scale_parameter_estimates, loc_avg, samples_avg, samples_nonavg = \
                self.encoder_transformed([image_input, *transformations])
            z_non_avg = tf.keras.layers.Concatenate(-1)(samples_nonavg)
            z = tf.keras.layers.Concatenate(-1)(samples_avg)
            z_loc_anchored = tf.keras.layers.Concatenate(-1)(loc_avg)
            z_loc = tf.keras.layers.Concatenate(-1)(loc_parameter_estimates)

            # Reconstruction
            reconstruction = self.decoder_labeled(z)
            # Calculate reconstruction loss (SAME)
            reconstruction_loss = tf.reduce_mean(self.reconstruction_loss(image_input, reconstruction))

            # Calculate KL and equivariance loss (SAME)
            kl_loss = tf.reduce_mean(self.kl_loss_function(loc_parameter_estimates, scale_parameter_estimates))

            if self.anchor_locations:
                # Calculate loss between the loc parameters and the transformed anchor
                equivariance_loss = tf.reduce_mean(self.equivariance_loss_function(z_loc, z_loc_anchored))
            else:
                # Calculate loss between the sampled latent and the sampled transformed anchor
                equivariance_loss = tf.reduce_mean(self.equivariance_loss_function(z_non_avg, z))

            # Equivariance loss
            self.equivariance_tracker.update_state(equivariance_loss)
            # Total loss
            self.calculate_loss_and_grads(reconstruction_loss, kl_loss, equivariance_loss, tape=tape)
            output_dictionary = dict(loss=self.total_loss_tracker.result(),
                                     reconstruction_loss_s=self.reconstruction_loss_tracker.result(),
                                     kl_loss_s=self.kl_loss_tracker.result(),
                                     equivariance_loss_s=self.equivariance_tracker.result())

            return output_dictionary

    def call(self, inputs, training=None, mask=None):
        return self.encoder_multiview(inputs)


class LSBDVAE(tf.keras.Model):
    """
    LSBDVAE Variational Autoencoder which creates equivariant transformations. The model can be trained in
    semi-supervised fashion.
    """

    def call(self, inputs, training=None, mask=None):
        return self.u_lsbd.call(inputs)

    def get_config(self):
        pass

    def __init__(self, encoder_backbones: List[tf.keras.models.Model], decoder_backbone: tf.keras.models.Model,
                 latent_spaces: List[LatentSpace], n_transforms: int, input_shape: Tuple[int, int, int],
                 reconstruction_loss=gaussian_loss, stop_gradient: bool = False, anchor_locations: bool = False,
                 **kwargs):
        super(LSBDVAE, self).__init__(**kwargs)

        self.u_lsbd = UnsupervisedLSBDVAE(encoder_backbones, decoder_backbone,
                                          latent_spaces, input_shape,
                                          reconstruction_loss, stop_gradient, **kwargs)
        self.s_lsbd = SupervisedLSBDVAE(encoder_backbones, decoder_backbone,
                                        latent_spaces, n_transforms, input_shape,
                                        reconstruction_loss, stop_gradient, anchor_locations, **kwargs)

    def fit_semi_supervised(self, x_l, x_l_transformations, x_u, epochs, batch_size: int = 32,
                            callback_list: Optional[List[tf.keras.callbacks.Callback]] = None, verbose=1) -> None:
        # assume x_l has shape (n_batches, n_transformed_datapoints, *data_shape):
        n_transformed_datapoints = x_l.shape[1]
        batch_size_l = batch_size // n_transformed_datapoints  # // to ensure integer batch size

        total_epochs = 0  # Total number of epochs = labeled epochs + unlabeled epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            if len(x_l) > 0:  # don't train if there is no labelled data
                print("Labelled training")
                self.s_lsbd.fit({"images": x_l, "transformations": x_l_transformations},
                                batch_size=batch_size_l, epochs=total_epochs + 1, callbacks=callback_list,
                                initial_epoch=total_epochs, verbose=verbose)
                total_epochs += 1
            if len(x_u) > 0:  # don't train if there is no unlabelled data
                print("Unlabelled training")
                self.u_lsbd.fit({"images": x_u}, batch_size=batch_size, epochs=total_epochs + 1,
                                callbacks=callback_list,
                                initial_epoch=total_epochs, verbose=verbose)
                total_epochs += 1
