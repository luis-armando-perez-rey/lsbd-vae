import os
import sys

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

from ood_generalisation.experiments import main_ood
from ood_generalisation.modules import presets

FACTOR_RANGES_DSPRITES_LIST = [
    presets.FACTOR_RANGES_DSPRITES_RTE,
    presets.FACTOR_RANGES_DSPRITES_RTR_POSX,
    presets.FACTOR_RANGES_DSPRITES_RTR_SCALE,
    presets.FACTOR_RANGES_DSPRITES_RTR_ROTATION,
    presets.FACTOR_RANGES_DSPRITES_EXTR_016,
    presets.FACTOR_RANGES_DSPRITES_EXTR_025,
    presets.FACTOR_RANGES_DSPRITES_EXTR_050,
    presets.FACTOR_RANGES_DSPRITES_EXTR_075,
]

if __name__ == "__main__":
    for factor_ranges in FACTOR_RANGES_DSPRITES_LIST:
        kwargs_lsbdvae_ = {
            # "data_parameters": presets.SQUARE_PARAMETERS, "use_angles_for_selection": True,
            # "data_parameters": presets.ARROW_PARAMETERS, "use_angles_for_selection": True,
            "data_parameters": {"data": "dsprites"}, "use_angles_for_selection": False,
            "factor_ranges": factor_ranges,
            "epochs": 5,
            "batch_size": 128,
            "architecture": "dense",  # "dense", "conv"
            "log_t_limit": (-30, -29),
            "correct_dsprites_symmetries": True,
        }
        main_ood.main(kwargs_lsbdvae_)
        print("Done!")
