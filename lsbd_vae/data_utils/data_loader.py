import numpy as np
import os
from PIL import Image
import h5py
import tensorflow as tf
from lsbd_vae.data_utils.transform_image import TransformImage
from lsbd_vae.data_utils.factor_dataset import FactorImageDataset


def load_factor_data(data, root_path=None, **kwargs) -> FactorImageDataset:
    options_dict = {
        "modelnet_colors": get_modelnet_colors,
        "pixel": get_transformed_pixel,
        "arrow": get_transformed_arrow,
        "modelnet40": get_h5_saved_data,
        "coil100": get_coil100,
    }
    return options_dict[data](root_path, **kwargs)


# noinspection PyUnusedLocal
def get_transformed_pixel(root_path: str, height: int = 32, width: int = 32, square_size: int = 1, **kwargs_transform):
    """
    Returns a TransformImage object created from a square of pixels
    :param root_path: not used in this function
    :param height: height of the image
    :param width: width of the image
    :param square_size: size of the square in pixels
    :param kwargs_transform: extra arguments passed to the TransformImage constructor
    :return:
    """
    pixel_img = np.zeros((height, width, 1))
    pixel_img[0:square_size, 0:square_size, 0] = 1
    return TransformImage(pixel_img, **kwargs_transform)


def get_transformed_arrow(root_path, arrow_size=32, **kwargs_transform):
    assert arrow_size in [32, 64, 128], f"arrow size {arrow_size} not supported"
    image_rgba = Image.open(os.path.join(root_path, "data", "arrow_images", f"arrow_{arrow_size}.png"))
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
        normalize: whether data should be in the range [0,1] (True) or [0, 255] (False).
        collection_list: list of collections (object types) to load
        data_type: type of data to load
        dataset_directory: directory where the data is stored

    Returns:
        FactorImageDataset object

    """
    # Add ModelNet classes`
    available_collections = ["airplane",
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
    available_data_types = ["train", "test"]
    image_list = []
    views_list = []
    labels_list = []
    ids_list = []
    for num_collection, collection in enumerate(collection_list):
        assert collection in available_collections, f"collection_list = {collection} is not available. Possible" \
                                                    f" values are {available_collections}"
        assert data_type in available_data_types, f"data_type = {data_type} is not available. Possible values " \
                                                  f"are {available_data_types}"

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
        except FileNotFoundError:
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
    ds, ds_info = tfds.load(
        'coil100',
        split='train',
        batch_size=-1,
        data_dir=data_path,
        with_info=True,
    )
    if ds_info.version == "1.0.0":
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
            object_data = file.get(object_type)
            for identity in object_data.keys():
                ids_data = object_data.get(identity)
                data = np.array(ids_data.get(data_type))
    return data
