import numpy as np
from typing import Optional
from scipy import ndimage
from matplotlib import colors

# local imports
from lsbd_vae.data_utils.factor_dataset import FactorImageDataset


class TransformImage(FactorImageDataset):
    """
    Transforms an image in various ways to create a dataset with factors

    Args:
        input_images: input image(s) that will be transformed, can be a single image or an array of images,
            i.e. should have shape (..., h, w, d)
        step_size_vert: step size of vertical shift in pixels, None means no vertical shift
        step_size_hor: step size of horizontal shift in pixels, None means no horizontal shift
        n_hues: number of hue changes, None means no hue change
        n_rotations: number of rotations, None means no rotations
        rotation_mode: mode for scipy.ndimage.rotate, default is "constant"
        rotation_cval: cval for scipy.ndimage.rotate, default is 0.0
        relative_hue_shift: whether to change the hue of every pixel relatively
            (as opposed to assigning the same hue to each pixel)
    """

    def __init__(self, input_images: np.ndarray,
                 step_size_vert: Optional[int] = None,
                 step_size_hor: Optional[int] = None,
                 n_hues: Optional[int] = None,
                 n_rotations: Optional[int] = None,
                 rotation_mode: str = "constant",
                 rotation_cval: float = 0.0,
                 relative_hue_shift: bool = False):
        assert len(input_images.shape) >= 3,\
            f"input_image must have shape (..., height, width, depth), but got {input_images.shape}"
        height, width, depth = input_images.shape[-3:]

        images = input_images
        images_dtype = images.dtype
        factor_values_list = []
        max_factor_values = []
        factor_names = []
        # 1. rotate images (must be done before translations, to ensure objects are rotated around their centre)
        if n_rotations is not None:
            assert n_rotations > 1, "n_rotations should be larger than 1"
            new_images = np.empty((n_rotations, *images.shape), dtype=images_dtype)
            for i in range(n_rotations):
                angle = i * 360 / n_rotations  # ndimage.rotate uses degrees, not radians
                new_images[i] = ndimage.rotate(images, angle, axes=(-2, -3), reshape=False,
                                               mode=rotation_mode, cval=rotation_cval)
            images = new_images
            # prepend factor values/name/max to factor_values_list/factor_names/max_factor_values
            factors_rotation = np.linspace(0, 2 * np.pi, num=n_rotations, endpoint=False)
            factor_values_list.insert(0, factors_rotation)
            max_factor_values.insert(0, 2 * np.pi)
            factor_names.insert(0, "rotation")

        # 2. hue shift
        if n_hues is not None:
            assert n_hues > 1, "n_hues should be larger than 1"
            assert depth == 3, "input_images must be RGB to perform hue shift"
            images_hsv = colors.rgb_to_hsv(images)
            new_images_hsv = np.empty((n_hues, *images.shape), dtype=images_dtype)
            for i in range(n_hues):
                hue = i / n_hues  # hues are given from 0 to 1
                new_images_hsv[i, ...] = images_hsv
                if relative_hue_shift:  # shift hue value of each pixel
                    new_images_hsv[i, ..., 0] += hue
                else:  # replace hue value for all pixels
                    new_images_hsv[i, ..., 0] = hue
            new_images_hsv[..., 0] = np.mod(new_images_hsv[..., 0], 1)  # wrap hue values so they are in [0,1]
            images = colors.hsv_to_rgb(new_images_hsv)
            # Re-normalize the image pixel values
            images = images - np.amin([0, np.amin(images)])
            images /= np.amax(images)

            # prepend factor values/name/max to factor_values_list/factor_names/max_factor_values
            factors_hue = np.linspace(0, 1, num=n_hues, endpoint=False)
            factor_values_list.insert(0, factors_hue)
            max_factor_values.insert(0, 1)
            factor_names.insert(0, "hue_shift")

        # 3. horizontal translation
        if step_size_hor is not None:
            assert step_size_hor > 0, "step_size_hor should be larger than 0"
            grid_size_hor = int(np.ceil(width / step_size_hor))
            new_images = np.empty((grid_size_hor, *images.shape), dtype=images_dtype)
            for pos_hor in range(grid_size_hor):
                new_images[pos_hor] = np.roll(images, pos_hor * step_size_hor, axis=-2)
            images = new_images
            # prepend factor values/name/max to factor_values_list/factor_names/max_factor_values
            factors_hor = np.arange(0, width, step_size_hor)
            factor_values_list.insert(0, factors_hor)
            max_factor_values.insert(0, width)
            factor_names.insert(0, "horizontal_translation")

        # 4. vertical translation
        if step_size_vert is not None:
            assert step_size_vert > 0, "step_size_vert should be larger than 0"
            grid_size_vert = int(np.ceil(height / step_size_vert))
            new_images = np.empty((grid_size_vert, *images.shape), dtype=images_dtype)
            for pos_vert in range(grid_size_vert):
                new_images[pos_vert] = np.roll(images, pos_vert * step_size_vert, axis=-3)
            images = new_images
            # prepend factor values/name/max to factor_values_list/factor_names/max_factor_values
            factors_vert = np.arange(0, height, step_size_vert)
            factor_values_list.insert(0, factors_vert)
            max_factor_values.insert(0, height)
            factor_names.insert(0, "vertical_translation")

        # set attributes (through superclass)
        super().__init__(images, factor_values_list, max_factor_values, factor_names)
