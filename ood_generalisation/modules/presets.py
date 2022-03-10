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
