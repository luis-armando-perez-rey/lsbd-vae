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

# for SO(2)xSO(2) datasets like Arrow & Square
FACTOR_RANGES_2D_1_16 = ((0, 0.5*np.pi), (0, 0.5*np.pi))
FACTOR_RANGES_2D_QUADRANT = ((0, np.pi), (0, np.pi))
FACTOR_RANGES_2D_9_16 = ((0, 1.5*np.pi), (0, 1.5*np.pi))

# for dsprites, which has factor shape (3, 6, 40, 32, 32), factors ["shape", "scale", "orientation", "x_pos", "y_pos"]
FACTOR_RANGES_DSPRITES_RTE = ((0, 2*np.pi / 3), (0, 2*np.pi*0.6), (0, 2*np.pi*120/360),
                              (0, 2*np.pi * 0.4), (0, 2*np.pi * 0.4))
FACTOR_RANGES_DSPRITES_RTR_POSX = ((0, 2*np.pi / 3), (0, 0), (0, 0),
                                   (0, 2*np.pi * 0.4), (0, 0))
FACTOR_RANGES_DSPRITES_RTR_SCALE = ((0, 2*np.pi / 3), (0, 2*np.pi*0.6), (0, 2*np.pi*120/360),
                                    (0, 2*np.pi * 0.4), (0, 2*np.pi * 0.4))
FACTOR_RANGES_DSPRITES_RTR_ROTATION = ((0, 2*np.pi / 3), (0, 2*np.pi*0.6), (0, 2*np.pi*120/360),
                                       (0, 2*np.pi * 0.4), (0, 2*np.pi * 0.4))
