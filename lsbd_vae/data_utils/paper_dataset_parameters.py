import os

# Load data
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

def return_data_parameters(data_name):
    if data_name == "arrow":
        data_parameters = {
            "root_path": PROJECT_PATH,
            "data": "arrow",
            "arrow_size": 64,
            "n_hues": 64,
            "n_rotations": 64,
        }
    elif data_name == "pixel4":
        data_parameters = {
            "data": "pixel",
            "height": 64,
            "width": 64,
            "step_size_vert": 1,
            "step_size_hor": 1,
            "square_size": 4
        }

    elif data_name == "pixel8":
        data_parameters = {
            "data": "pixel",
            "height": 64,
            "width": 64,
            "step_size_vert": 1,
            "step_size_hor": 1,
            "square_size": 8
        }
    elif data_name == "pixel16":
        data_parameters = {
            "data": "pixel",
            "height": 64,
            "width": 64,
            "step_size_vert": 1,
            "step_size_hor": 1,
            "square_size": 16
        }
    elif data_name == "modelnet":
        data_parameters = {
            "root_path": PROJECT_PATH,
            "dataset_filename": "modelnet_color_single_64_64.h5",
            "data": "modelnet_colors"
        }
    elif data_name == "modelnet40":
        data_parameters = {
            "root_path": "/data/aligned64",
            "data": "modelnet40",
            "collection_list": [
                "airplane"],
            "data_type": "train",
            "dataset_directory": ""
        }
    elif data_name == "r_modelnet40_train":

        print(os.path.join(PROJECT_PATH, "data/aligned64"))
        data_parameters = {
            "root_path": os.path.join(PROJECT_PATH, "data/aligned64"),
            "data": "modelnet40",
            "collection_list": [
                "airplane", "car", "chair"],
            "data_type": "train",
            "dataset_directory": ""
        }

    elif data_name == "r_modelnet40_test":
        data_parameters = {
            "root_path": os.path.join(PROJECT_PATH, "data/aligned64"),
            "data": "modelnet40",
            "collection_list": [
                "airplane", "car", "chair"],
            "data_type": "test",
            "dataset_directory": ""
        }
    elif data_name == "complete_modelnet40_train":
        data_parameters = {
            "root_path": os.path.join(PROJECT_PATH, "data/aligned64"),
            "data": "modelnet40",
            "collection_list": ["airplane",
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
                                "xbox"],
            "data_type": "train",
            "dataset_directory": ""
        }
    elif data_name == "complete_modelnet40_test":
        data_parameters = {
            "root_path": os.path.join(PROJECT_PATH, "data/aligned64"),
            "data": "modelnet40",
            "collection_list": ["airplane",
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
                                "xbox"],
            "data_type": "test",
            "dataset_directory": ""
        }


    elif data_name == "coil100":
        data_parameters = {
            "root_path": PROJECT_PATH,
            "data": "coil100",
        }
    elif data_name == "smallnorb":
        data_parameters = {
            "root_path": PROJECT_PATH,
            "data": "smallnorb"
        }
    else:
        data_parameters = None
    return data_parameters
