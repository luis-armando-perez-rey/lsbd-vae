from pathlib import Path
import pickle
import time
import os
import sys

# add project root to path
ROOT_PATH = os.path.dirname(os.path.dirname(os.getcwd()))  # lsbd-vae folder
sys.path.append(ROOT_PATH)

from ood_generalisation.modules import utils
from ood_generalisation.experiments import main_ood


def main(timestamp, neptune_run_id):
    use_neptune = True

    if use_neptune:
        import neptune.new as neptune
        from neptune_config import api_key
        neptune_project = "TUe/ood-lsbd"
        print(f"\n=== Logging to Neptune project {neptune_project}, existing run {neptune_run_id} "
              f"with timestamp {timestamp} ==")
        neptune_run_ = neptune.init(project=neptune_project, api_token=api_key.API_KEY, run=neptune_run_id)

        save_path = Path("results", timestamp)
        settings_path = save_path / "settings.p"
        with open(settings_path, "rb") as f:
            print("... loading kwargs_lsbdvae from", settings_path)
            kwargs_lsbdvae = pickle.load(f)

        print("\n=== Experiment kwargs_lsbdvae: ===")
        for key, value in kwargs_lsbdvae.items():
            print(key, "=", value)

        print("\n=== Start evaluation ===")
        start_time = time.time()
        main_ood.run_lsbdvae(save_path=save_path, neptune_run=neptune_run_, **kwargs_lsbdvae)
        print("\n=== Evaluation done ===")
        end_time = time.time()
        utils.print_and_log_time(start_time, end_time, neptune_run_, "time_elapsed/evaluate")
        print()
        if neptune_run_ is not None:
            neptune_run_.stop()


TIMESTAMPS_AND_NEPTUNE_RUN_IDS = [
    ("2022-03-27-11-21-00", "OOD-213"),
]

if __name__ == "__main__":
    for (timestamp_, neptune_run_id_) in TIMESTAMPS_AND_NEPTUNE_RUN_IDS:
        main(timestamp_, neptune_run_id_)
        print("Done!")
