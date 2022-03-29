import os
import sys

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

from ood_generalisation.experiments import main_ood
from ood_generalisation.modules import presets


FACTOR_RANGES_DSPRITES_LIST_LONG = [
    presets.FACTOR_RANGES_DSPRITES_RTE,
    presets.FACTOR_RANGES_DSPRITES_RTR_POSX,
    presets.FACTOR_RANGES_DSPRITES_RTR_SCALE,
    presets.FACTOR_RANGES_DSPRITES_RTR_ROTATION,
    presets.FACTOR_RANGES_DSPRITES_EXTR_016,
    presets.FACTOR_RANGES_DSPRITES_EXTR_025,
    presets.FACTOR_RANGES_DSPRITES_EXTR_050,
    presets.FACTOR_RANGES_DSPRITES_EXTR_075,
]


FACTOR_RANGES_2D_LIST = [
    None,
    presets.FACTOR_RANGES_2D_1_16,
    presets.FACTOR_RANGES_2D_QUADRANT,
    presets.FACTOR_RANGES_2D_9_16,
]

FACTOR_RANGES_DSPRITES_LIST = [
    None,
    presets.FACTOR_RANGES_DSPRITES_RTE,
    presets.FACTOR_RANGES_DSPRITES_RTR_POSX,
    presets.FACTOR_RANGES_DSPRITES_EXTR_050,
]

FACTOR_RANGES_SHAPES3D_LIST = [
    None,
    presets.FACTOR_RANGES_SHAPES3D_RTE,
    presets.FACTOR_RANGES_SHAPES3D_RTR,
    presets.FACTOR_RANGES_SHAPES3D_EXTR,
]

# FULL EXPERIMENTS (but this takes longer than 24 hours)
DATA_PARAMS_LIST = [presets.SQUARE_PARAMETERS, presets.ARROW_PARAMETERS, {"data": "dsprites"}, {"data": "shapes3d"}]
USE_ANGLES_LIST = [True, True, False, False]
FACTOR_RANGES_LISTOFLISTS =\
    [FACTOR_RANGES_2D_LIST, FACTOR_RANGES_2D_LIST, FACTOR_RANGES_DSPRITES_LIST, FACTOR_RANGES_SHAPES3D_LIST]

n_repetitions = 3


if __name__ == "__main__":
    # for data_parameters, use_angles, factor_ranges_list in zip(DATA_PARAMS_LIST, USE_ANGLES_LIST, FACTOR_RANGES_LISTOFLISTS):
    #     for factor_ranges in factor_ranges_list:
    for factor_ranges in [presets.FACTOR_RANGES_DSPRITES_RTR_POSX, presets.FACTOR_RANGES_DSPRITES_EXTR_050]:
        for _ in range(n_repetitions):
            kwargs_lsbdvae_ = {
                "data_parameters": {"data": "dsprites"}, "use_angles_for_selection": False,
                "factor_ranges": factor_ranges,
                "epochs": 200,
                "batch_size": 8,
                "supervision_batch_size": 32,
                "architecture": "dislib",  # "dense", "conv", "dislib"
                "reconstruction_loss": "bernoulli",  # "gaussian", "bernoulli"
                "log_t_limit": (-10, -6),
                "correct_dsprites_symmetries": True,
                "early_stopping": True,
            }
            main_ood.main(kwargs_lsbdvae_)
            print("Done!")
