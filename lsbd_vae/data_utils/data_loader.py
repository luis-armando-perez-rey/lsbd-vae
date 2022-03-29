import numpy as np
import os
from PIL import Image
import h5py
import json
import tensorflow as tf

from typing import Type

from lsbd_vae.data_utils.factor_dataset import FactorImageDataset
from lsbd_vae.data_utils.transform_image import TransformImage


def load_factor_data(data, root_path=None, **kwargs) -> Type[FactorImageDataset]:
    options_dict = {
        "sim_toy": get_sim_toy,
        "dsprites": get_dsprites,
        "shapes3d": get_shapes3d,
        "modelnet_colors": get_modelnet_colors,
        "pixel": get_transformed_pixel,
        "arrow": get_transformed_arrow,
        "transformed_image": get_transformed_image,
        "modelnet40": get_h5_saved_data,
        "modelnet_cars": get_modelnet_cars,
        "coil100": get_coil100,
        "smallnorb": get_smallnorb,
    }
    return options_dict[data](root_path, **kwargs)


def get_sim_toy(root_path):
    assert root_path is not None, "project root path is not supplied"
    filepath = os.path.join(root_path, "data", "sim_toy_ordered", "sim_toy_np_ordered.npz")
    with np.load(filepath, mmap_mode="r") as f:
        images = f["images"]
    images = images.astype('float32') / 255.
    images = images.reshape((4, 4, 2, 3, 3, 40, 40, 64, 64, 3))
    factor_names = ["object_color", "object_shape", "object_size", "camera_height", "background_color",
                    "horizontal_axis", "vertical_axis"]
    return FactorImageDataset(images, factor_names=factor_names)


def get_dsprites(root_path, noncyclic_gap=0.1):
    """
    Args:
        root_path: path where the data folder is
        noncyclic_gap: for assigning angular values to the factor values, this represents the percentage of a circle
            that will not have any factor values assigned to it, for factors that are not cyclic. E.g. if factor values
            are from 0 to 0.9, and the noncyclic_gap is 0.1, the max_factor_value will be 1.
    """
    assert root_path is not None, "project root path is not supplied"
    dsprites_path = os.path.join(root_path, "data", "dsprites", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    with np.load(dsprites_path, encoding="latin1", allow_pickle=True) as f:
        images = f["imgs"]  # np array of shape (737280, 64, 64) containing 0's and 1's (dtype uint8)
        factor_values_dict = f["metadata"].item()["latents_possible_values"]
    images = images.reshape((3, 6, 40, 32, 32, 64, 64, 1)).astype("float32")
    factor_names = ["shape", "scale", "orientation", "x_pos", "y_pos"]
    factor_values_list = [
        factor_values_dict["shape"] - 1,  # values are given as 1, 2, 3, subtract 1 to start from 0. Factor values are
                                          # then: 0=square, 1=ellips, 2=heart
        (factor_values_dict["scale"] - 0.5) * 2,  # values are given in [0.5, 1.], rescale to be in [0., 1.]
        np.mod(factor_values_dict["orientation"], 2*np.pi),  # last value is 2*pi, so should be converted to 0
        factor_values_dict["posX"],  # values from 0.0 to 1.0
        factor_values_dict["posY"],  # values from 0.0 to 1.0
    ]
    max_factor_values = [
        3,
        1 / (1 - noncyclic_gap),
        2 * np.pi,
        1 / (1 - noncyclic_gap),
        1 / (1 - noncyclic_gap)
    ]
    return FactorImageDataset(images, max_factor_values=max_factor_values, factor_names=factor_names,
                              factor_values_list=factor_values_list)


def get_shapes3d(root_path, noncyclic_gap=0.1):
    """
    Args:
        root_path: path where the data folder is
        noncyclic_gap: for assigning angular values to the factor values, this represents the percentage of a circle
            that will not have any factor values assigned to it, for factors that are not cyclic. E.g. if factor values
            are from 0 to 0.9, and the noncyclic_gap is 0.1, the max_factor_value will be 1.
    """
    assert root_path is not None, "project root path is not supplied"
    shapes3d_path = os.path.join(root_path, "data", "shapes3d", "3dshapes.h5")
    dataset = h5py.File(shapes3d_path, 'r')
    images = dataset['images'][()]  # numpy array shape [480000,64,64,3], uint8 in range(256)
    # labels = dataset['labels'][()]  # numpy array shape [480000,6], float64

    images = images.reshape((10, 10, 10, 8, 4, 15, 64, 64, 3)).astype("float32") / 255

    factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    # 0. floor_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
    # 1. wall_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
    # 2. object_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
    # 3. scale: 8 values from 0.75 to 1.25, instead use values from 0 to 1
    # 4. shape: 4 values from 0 to 3, 0=cube, 1=cylinder, 2=ball, 3=oblong
    # 5. orientation: 15 values from -30 to 30, instead use values from 0 to 1
    factor_values_list = [
        np.linspace(0, 0.9, num=10),
        np.linspace(0, 0.9, num=10),
        np.linspace(0, 0.9, num=10),
        np.linspace(0, 1, num=8),
        np.linspace(0, 3, num=4),
        np.linspace(0, 1, num=15),
    ]
    max_factor_values = [
        1,
        1,
        1,
        1 / (1 - noncyclic_gap),
        4,
        1 / (1 - noncyclic_gap)
    ]
    return FactorImageDataset(images, max_factor_values=max_factor_values, factor_names=factor_names,
                              factor_values_list=factor_values_list)


def get_modelnet_cars(root_path,
                      resolution=(64, 64),
                      intensities=np.arange(0, 6),
                      colors=np.arange(0, 6),
                      azimuths=np.arange(0, 6),
                      elevations=np.arange(0, 6),
                      locations=np.arange(0, 6)
                      ):
    factor_names = ["identity", "color", "azimuth", "elevation", "light_location", "intensity"]
    print("Start loading of ModelNet cars dataset ...")
    total_factors = 6
    filenames = os.listdir(root_path)
    unique_ids = np.unique([filename.split("_")[1] for filename in filenames])
    factor_values = np.zeros(
        (len(unique_ids), len(colors), len(azimuths), len(elevations), len(locations), len(intensities), total_factors),
        dtype=int)
    images = np.zeros((len(unique_ids), len(colors), len(azimuths), len(elevations), len(locations), len(intensities),
                       resolution[0], resolution[1], 3))
    print("Images shape", images.shape)
    # Load the JSON file with the car labels
    with open(os.path.join(os.path.dirname(root_path), "labels_per_car.json")) as f:
        labels_dictionary = json.load(f)
    labels = np.zeros((len(unique_ids), len(colors), len(azimuths), len(elevations), len(locations), len(intensities)),
                      dtype=int)
    for num_id, unique_id in enumerate(unique_ids):
        for num_intensity, intensity in enumerate(intensities):
            for num_color, color in enumerate(colors):
                for num_azimuth, azimuth in enumerate(azimuths):
                    for num_elevation, elevation in enumerate(elevations):
                        for num_location, location in enumerate(locations):
                            labels[num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity] = \
                                labels_dictionary[unique_id]
                            filename = "car_" + unique_id + "_" \
                                       + str(color) + "_" \
                                       + str(azimuth) + "_" \
                                       + str(elevation + 1) + "_" \
                                       + str(location) + "_" \
                                       + str(intensity) + ".png"
                            filepath = os.path.join(root_path, filename)
                            images[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity] = np.array(
                                Image.open(filepath).resize(resolution))[:, :, :3] / 255.0
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 0] = int(
                                unique_id)
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 1] = int(
                                color)
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 2] = int(
                                azimuth)
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 3] = int(
                                elevation)
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 4] = int(
                                location)
                            factor_values[
                                num_id, num_color, num_azimuth, num_elevation, num_location, num_intensity, 5] = int(
                                intensity)
    return FactorImageDataset(images, factor_names=factor_names, labels=np.array(labels))


def get_transformed_pixel(root_path, height=32, width=32, square_size=1, **kwargs_transform):
    pixel_img = np.zeros((height, width, 1))
    pixel_img[0:square_size, 0:square_size, 0] = 1
    return TransformImage(pixel_img, **kwargs_transform)


def get_transformed_image(root_path, image, **kwargs_transform):
    return TransformImage(image, **kwargs_transform)


def get_transformed_arrow(root_path, arrow_size=32, **kwargs_transform):
    assert arrow_size in [32, 64, 128], "arrow size not supported"
    image_rgba = Image.open(os.path.join(root_path, "data", "single_images", f"arrow_{arrow_size}.png"))
    image_rgb = image_rgba.convert("RGB")
    arrow_img = np.asarray(image_rgb)
    arrow_img = arrow_img.astype('float32') / 255.
    return TransformImage(arrow_img, **kwargs_transform)


def get_modelnet_colors(root_path, dataset_filename, object_type=None, normalize=True):
    """
    Returns a TransformImage object created from ModelNet40 dataset of objects with periodic colors and rotated
    Args:
        root_path: path to the root of the project
        dataset_filename: filename of the .h5 data to be loaded
        object_type: type of object saved in the data file
        normalize: whether data should be in the range [0,1] (True) or [0, 255] (False).

    Returns:
        FactorImageDataset object
    """
    dataset_filepath = os.path.join(root_path, "data", "modelnet40", dataset_filename)
    # Read the images
    images = read_modelnet_data_h5(dataset_filepath, object_type, "images")
    if normalize:
        images = images.astype('float32') / np.amax(images)

    # Read the factors
    colors = read_modelnet_data_h5(dataset_filepath, object_type, "colors")
    views = read_modelnet_data_h5(dataset_filepath, object_type, "views")

    # Convert integer range to angular range
    unique_angle_colors = np.unique(colors)
    unique_views = np.unique(views)
    # Create FactorImageDataset lists
    factor_values = [unique_angle_colors, unique_views]
    max_factor_values = [np.amax(factor) for factor in factor_values]

    return FactorImageDataset(images=images,
                              factor_values_list=factor_values,
                              max_factor_values=max_factor_values,
                              factor_names=["colors_angle", "rotation_angle"])


def get_h5_saved_data(root_path, collection_list, data_type, dataset_directory, normalize=True):
    """
    Returns a TransformImage object created from ModelNet40 dataset of objects with periodic colors and rotated
    Args:
        root_path: path to the root of the project
        dataset_filename: filename of the .h5 data to be loaded
        object_type: type of object saved in the data file
        normalize: whether data should be in the range [0,1] (True) or [0, 255] (False).

    Returns:
        FactorImageDataset object
    """
    AVAILABLE_COLLECTIONS = ["Shape", "Culture"]
    # Add ModelNet classes`
    AVAILABLE_COLLECTIONS += ["airplane",
                              "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
                              "car", "chair", "cone", "cup", "curtain",
                              "desk", "door", "dresser",
                              "flower_pot",
                              "glass_box",
                              "guitar",
                              "keyboard",
                              "lamp", "laptop",
                              "mantel", "monitor",
                              "night_stand",
                              "person", "piano", "plant",
                              "radio", "range_hood",
                              "sink", "sofa", "stairs", "stool",
                              "table", "tent", "toilet", "tv_stand",
                              "vase",
                              "wardrobe",
                              "xbox"]
    AVAILABLE_DATA_TYPES = ["train", "test"]
    image_list = []
    views_list = []
    labels_list = []
    ids_list = []
    for num_collection, collection in enumerate(collection_list):
        assert collection in AVAILABLE_COLLECTIONS, "collection_list = {} is not available. Possible values are {}".format(
            collection, AVAILABLE_COLLECTIONS)
        assert data_type in AVAILABLE_DATA_TYPES, "data_type = {} is not available. Possible values are {}".format(
            data_type, AVAILABLE_DATA_TYPES)

        dataset_filename = collection + "_" + data_type + ".h5"
        print(f"Loading file {num_collection} ", dataset_filename)
        dataset_filepath = os.path.join(root_path, dataset_directory, dataset_filename)
        # Read the images
        images = read_data_h5(dataset_filepath, "images")
        if normalize:
            images = images.astype('float32') / np.amax(images)
        image_list.append(images)
        # Read the rotations
        views = read_data_h5(dataset_filepath, "views")
        views_list.append(views)
        # Read category integer class_labels
        try:
            labels = read_data_h5(dataset_filepath, "class_int")
        except:
            print("No labels detected in the dataset")
            labels = None
        labels_list.append(labels)
        ids = read_data_h5(dataset_filepath, "ids")
        ids_list.append(ids)

    images = np.concatenate(image_list, axis=0)
    # Unique id for each image just enumerate all images
    # ids = np.concatenate(ids_list, axis=0)
    ids = np.arange(len(images))
    labels = np.concatenate(labels_list, axis=0)
    views = np.concatenate(views_list, axis=0)
    # Convert integer range to angular range
    unique_views = np.unique(views)
    unique_ids = np.unique(ids)
    # Create FactorImageDataset lists
    factor_values = [unique_ids, unique_views]
    max_factor_values = [np.amax(factor) for factor in factor_values]
    return FactorImageDataset(images=images,
                              factor_values_list=factor_values,
                              max_factor_values=max_factor_values,
                              factor_names=["object_ids", "rotation_angle"],
                              labels=labels)


def get_coil100(root_path, rescale_size=64):
    import tensorflow_datasets as tfds
    # load dataset from tfds
    data_path = os.path.join(root_path, "data", "tfds_coil100")
    ds = tfds.load(
        'coil100',
        split='train',
        batch_size=-1,
        data_dir=data_path
    )

    if tfds.__version__ == "1.2.0":
        angles = tfds.as_numpy(ds["label"])
    else:
        angles = tfds.as_numpy(ds["angle_label"])
    obj_ids = tfds.as_numpy(ds["object_id"])

    # rescale if needed (set rescale_size=None to skip)
    if rescale_size is not None:
        images = tfds.as_numpy(tf.image.resize(ds["image"], size=(rescale_size, rescale_size)))
    else:
        images = tfds.as_numpy(ds["image"])

    # sort by angle and obj_id, then reshape to factor shape
    ind = np.lexsort((angles, obj_ids))
    images = images[ind]
    images = np.reshape(images, (100, 72, *images.shape[1:]))

    # sort and reshape obj_ids too to use as labels
    obj_ids_sorted = obj_ids[ind]
    obj_ids_sorted = [int(str(class_bytes).replace("b\'obj", "").replace("\'", "")) for class_bytes in obj_ids_sorted]
    obj_ids_factor = np.reshape(obj_ids_sorted, (100, 72))

    # normalize
    images = images / 255.

    return FactorImageDataset(images, factor_names=["obj_id", "angle"], labels=obj_ids_factor)


def get_smallnorb(root_path, angles_only=True, rescale_size=64):
    import tensorflow_datasets as tfds
    from skimage.transform import resize
    # load dataset from tfds
    data_path = os.path.join(root_path, "data", "tfds_smallnorb")
    ds_np = tfds.as_numpy(tfds.load(
        'smallnorb',
        split=['train', 'test'],
        batch_size=-1,
        data_dir=data_path
    ))
    x_train, x_test = ds_np
    image = np.concatenate([x_train["image"], x_test["image"]], axis=0)
    image2 = np.concatenate([x_train["image2"], x_test["image2"]], axis=0)

    label_azimuth = np.concatenate([x_train["label_azimuth"], x_test["label_azimuth"]], axis=0)
    instance = np.concatenate([x_train["instance"], x_test["instance"]], axis=0)
    label_category = np.concatenate([x_train["label_category"], x_test["label_category"]], axis=0)
    label_elevation = np.concatenate([x_train["label_elevation"], x_test["label_elevation"]], axis=0)
    label_lighting = np.concatenate([x_train["label_lighting"], x_test["label_lighting"]], axis=0)

    ind = np.lexsort((label_azimuth, instance, label_category, label_elevation, label_lighting))
    image_sorted = image[ind]
    image2_sorted = image2[ind]
    image_factor = np.reshape(image_sorted, (6, 9, 5, 10, 18, 96, 96, 1))
    image2_factor = np.reshape(image2_sorted, (6, 9, 5, 10, 18, 96, 96, 1))
    images_factor = np.stack([image_factor, image2_factor], axis=0)  # shape (2, 6, 9, 5, 10, 18, 96, 96, 1)

    if angles_only:
        images_factor = np.reshape(images_factor, (2 * 6 * 9 * 5 * 10, 18, 96, 96, 1))
        factor_names = ["other", "angle"]
    else:
        factor_names = ["camera", "lighting", "elevation", "category", "instance", "angle"]

    # normalize
    images_factor = images_factor.astype('float32') / 255.

    # rescale if needed (set rescale_size=None to skip)
    if rescale_size is not None:
        images_factor = resize(images_factor, images_factor.shape[:-3] + (rescale_size, rescale_size, 1))

    return FactorImageDataset(images_factor, factor_names=factor_names)


def read_data_h5(data_filepath, data_type):
    """
    Read data from h5 file with 1 level of hierarchy
    Args:
        data_filepath: path to the .h5 file
        data_type: key value from which data is read

    Returns:

    """
    with h5py.File(data_filepath, "r") as file:
        data = np.array(file.get(data_type))
    return data


# noinspection PyBroadException
def read_modelnet_data_h5(data_filepath, object_type, data_type):
    """
    Args:
        data_filepath: path to the .h5 file from which data is loaded
        object_type: if None return all object types.
        data_type: data types available are images, colors and views
    Returns:
    """
    with h5py.File(data_filepath, "r") as file:
        # Get the data
        if object_type is None:
            for object_ in file.keys():
                object_data = file.get(object_)
                for identity in object_data.keys():
                    ids_data = object_data.get(identity)
                    data = np.array(ids_data.get(data_type))
        else:
            try:
                object_data = file.get(object_type)
                for identity in object_data.keys():
                    ids_data = object_data.get(identity)
                    data = np.array(ids_data.get(data_type))
            except:
                print(
                    f"Data with object type: {object_type} and data type {data_type} is not available in {data_filepath}")
                data = None
    return data
