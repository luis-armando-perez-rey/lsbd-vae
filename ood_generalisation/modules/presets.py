import numpy as np

SQUARE_PARAMETERS = {
    "data": "pixel",
    "height": 64,
    "width": 64,
    "step_size_vert": 1,
    "step_size_hor": 1,
    "square_size": 16
}

ARROW_PARAMETERS = {
    "data": "arrow",
    "arrow_size": 64,
    "n_hues": 64,
    "n_rotations": 64,
}

# for SO(2)xSO(2) datasets like Arrow & Square, with use_angles_for_selection=True
FACTOR_RANGES_2D_1_16 = ((0, 0.5*np.pi), (0, 0.5*np.pi))
FACTOR_RANGES_2D_QUADRANT = ((0, np.pi), (0, np.pi))
FACTOR_RANGES_2D_9_16 = ((0, 1.5*np.pi), (0, 1.5*np.pi))

# for dsprites, which has factor shape (3, 6, 40, 32, 32), factors ["shape", "scale", "orientation", "x_pos", "y_pos"]
#   shape: [0., 1., 2.] (0=square, 1=ellips, 2=heart)
#   scale: [0. , 0.2, 0.4, 0.6, 0.8, 1. ]
#   orientation: 40 values 0 to 2*pi (last value is modulo'd to 0)
#   x_pos: 32 values from 0.0 to 1.0
#   y_pos: 32 values from 0.0 to 1.0
# to be used with use_angles_for_selection=False
FACTOR_RANGES_DSPRITES_RTE = ((1.0, 1.1), (0.0, 0.6), (2*np.pi*120/360, 2*np.pi*240/360), (0.6, 1.1), (0.6, 1.1))
FACTOR_RANGES_DSPRITES_RTR_POSX = ((0.0, 0.1), (0.0, 1.1), (0.0, 2*np.pi), (0.5, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_RTR_SCALE = ((0.0, 0.1), (0.5, 1.1), (0.0, 2*np.pi), (0.0, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_RTR_ROTATION = ((0.0, 0.1), (0.0, 1.1), (2*np.pi*90/360, 2*np.pi), (0.0, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_EXTR_016 = ((0.0, 3.0), (0.0, 1.1), (0.0, 2*np.pi), (0.16, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_EXTR_025 = ((0.0, 3.0), (0.0, 1.1), (0.0, 2*np.pi), (0.25, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_EXTR_050 = ((0.0, 3.0), (0.0, 1.1), (0.0, 2*np.pi), (0.50, 1.1), (0.0, 1.1))
FACTOR_RANGES_DSPRITES_EXTR_075 = ((0.0, 3.0), (0.0, 1.1), (0.0, 2*np.pi), (0.75, 1.1), (0.0, 1.1))

# for shapes3d, which has factor shape (10, 10, 10, 8, 4, 15),
#   factors ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
# 0. floor_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
# 1. wall_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
# 2. object_hue: 10 values from 0 to 0.9 (basically cyclic, first is red, last is purple)
# 3. scale: 8 values from  0 to 1
# 4. shape: 4 values from 0 to 3, 0=cube, 1=cylinder, 2=ball, 3=oblong
# 5. orientation: 15 values from 0 to 1
FACTOR_RANGES_SHAPES3D_RTE = ((0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (1.0, 1.1), (3.0, 3.1), (0.0, 0.01))
FACTOR_RANGES_SHAPES3D_RTR = ((0.0, 1.0), (0.0, 1.0), (0.5, 1.0), (0.0, 1.1), (3.0, 3.1), (0.0, 1.1))
FACTOR_RANGES_SHAPES3D_EXTR = ((0.5, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.1), (0.0, 3.1), (0.0, 1.1))

# load data_paramaters, factor ranges, and use_angles_for_selection given a single name (as used in dislib)
DATA_RANGES_ANGLES = {
    "arrow": (ARROW_PARAMETERS, None, True),
    "square": (SQUARE_PARAMETERS, None, True),
    "dsprites_lsbd": ({"data": "dsprites"}, None, False),
    "shapes3d_lsbd": ({"data": "shapes3d"}, None, False),
    "arrow_1_16": (ARROW_PARAMETERS, FACTOR_RANGES_2D_1_16, True),
    "arrow_quadrant": (ARROW_PARAMETERS, FACTOR_RANGES_2D_QUADRANT, True),
    "arrow_9_16": (ARROW_PARAMETERS, FACTOR_RANGES_2D_9_16, True),
    "square_1_16": (SQUARE_PARAMETERS, FACTOR_RANGES_2D_1_16, True),
    "square_quadrant": (SQUARE_PARAMETERS, FACTOR_RANGES_2D_QUADRANT, True),
    "square_9_16": (SQUARE_PARAMETERS, FACTOR_RANGES_2D_9_16, True),
    "dsprites_rte": ({"data": "dsprites"}, FACTOR_RANGES_DSPRITES_RTE, False),
    "dsprites_rtr": ({"data": "dsprites"}, FACTOR_RANGES_DSPRITES_RTR_POSX, False),
    "dsprites_extr": ({"data": "dsprites"}, FACTOR_RANGES_DSPRITES_EXTR_050, False),
    "shapes3d_rte": ({"data": "shapes3d"}, FACTOR_RANGES_SHAPES3D_RTE, False),
    "shapes3d_rtr": ({"data": "shapes3d"}, FACTOR_RANGES_SHAPES3D_RTR, False),
    "shapes3d_extr": ({"data": "shapes3d"}, FACTOR_RANGES_SHAPES3D_EXTR, False),
}

DATASET_INTERESTING_LATENT_DIMS = {
    "arrow": (0, 1),
    "square": (0, 1),
    "dsprites_lsbd": (3, 4),
    "shapes3d_lsbd": (0, 1),
    "arrow_1_16": (0, 1),
    "arrow_quadrant": (0, 1),
    "arrow_9_16": (0, 1),
    "square_1_16": (0, 1),
    "square_quadrant": (0, 1),
    "square_9_16": (0, 1),
    "dsprites_rte": (3, 4),
    "dsprites_rtr": (0, 3),
    "dsprites_extr": (3, 4),
    "shapes3d_rte": (0, 1),
    "shapes3d_rtr": (2, 4),
    "shapes3d_extr": (0, 1),
}
