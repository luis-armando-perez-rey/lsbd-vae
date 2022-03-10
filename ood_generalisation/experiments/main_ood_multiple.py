import os
import sys

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

from ood_generalisation.experiments import main_ood

if __name__ == "__main__":
    for log_t_limit in [(-11, -10), (-12, -11), (-15, -14), (-20, -19), (-30, -29)]:
        kwargs_lsbdvae_ = {
            # "data_parameters": presets.SQUARE_PARAMETERS, "use_angles_for_selection": True,
            # "data_parameters": presets.ARROW_PARAMETERS, "use_angles_for_selection": True,
            "data_parameters": {"data": "dsprites"}, "use_angles_for_selection": False,
            "factor_ranges": None,
            "epochs": 10,
            "batch_size": 128,
            "architecture": "dense",  # "dense", "conv"
            "log_t_limit": log_t_limit,
            "correct_dsprites_symmetries": True,
        }
        main_ood.main(kwargs_lsbdvae_)
        print("Done!")
