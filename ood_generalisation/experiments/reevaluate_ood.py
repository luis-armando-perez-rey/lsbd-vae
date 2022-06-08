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




TIMESTAMPS_AND_NEPTUNE_RUN_IDS_ARROW = [
    ("2022-03-29-19-17-46", "OOD-237"),
    ("2022-03-29-19-14-51", "OOD-236"),
    ("2022-03-29-19-11-37", "OOD-235"),
    ("2022-03-29-19-07-54", "OOD-234"),
    ("2022-03-29-19-04-00", "OOD-233"),
    ("2022-03-29-18-59-35", "OOD-232"),
    ("2022-03-29-18-54-31", "OOD-231"),
    ("2022-03-29-18-49-04", "OOD-230"),
    ("2022-03-29-18-42-54", "OOD-229"),
    ("2022-03-29-18-36-11", "OOD-228"),
    ("2022-03-29-18-29-41", "OOD-227"),
    ("2022-03-29-18-22-21", "OOD-226"),
    ("2022-03-24-17-58-12", "OOD-188"),
    ("2022-03-24-17-55-17", "OOD-187"),
    ("2022-03-24-17-50-10", "OOD-186"),
    ("2022-03-24-17-44-00", "OOD-185"),
    ("2022-03-24-17-36-30", "OOD-184"),
    ("2022-03-24-17-31-28", "OOD-183"),
    ("2022-03-24-17-25-47", "OOD-182"),
    ("2022-03-24-17-20-28", "OOD-181"),
    ("2022-03-24-17-15-51", "OOD-180"),
    ("2022-03-24-17-12-36", "OOD-179"),
    ("2022-03-24-17-06-43", "OOD-178"),
    ("2022-03-24-17-02-57", "OOD-177"),
]

TIMESTAMPS_AND_NEPTUNE_RUN_IDS_SQUARE = [
    ("2022-03-29-18-20-00", "OOD-225"),
    ("2022-03-29-18-17-43", "OOD-224"),
    ("2022-03-29-18-15-26", "OOD-223"),
    ("2022-03-29-18-13-05", "OOD-222"),
    ("2022-03-29-18-10-22", "OOD-221"),
    ("2022-03-29-18-07-37", "OOD-220"),
    ("2022-03-29-18-04-48", "OOD-219"),
    ("2022-03-29-18-01-48", "OOD-218"),
    ("2022-03-29-17-58-45", "OOD-217"),
    ("2022-03-29-17-55-51", "OOD-216"),
    ("2022-03-29-17-53-03", "OOD-215"),
    ("2022-03-29-17-50-27", "OOD-214"),
    ("2022-03-24-17-00-47", "OOD-176"),
    ("2022-03-24-16-58-26", "OOD-175"),
    ("2022-03-24-16-56-02", "OOD-174"),
    ("2022-03-24-16-53-23", "OOD-173"),
    ("2022-03-24-16-50-19", "OOD-172"),
    ("2022-03-24-16-47-38", "OOD-171"),
    ("2022-03-24-16-44-06", "OOD-170"),
    ("2022-03-24-16-40-09", "OOD-169"),
    ("2022-03-24-16-35-31", "OOD-168"),
    ("2022-03-24-16-30-58", "OOD-167"),
    ("2022-03-24-16-28-25", "OOD-166"),
    ("2022-03-24-16-25-31", "OOD-165"),
]

TIMESTAMPS_AND_NEPTUNE_RUN_IDS_DSPRITES = [
    ("2022-03-27-11-21-00", "OOD-213"),
    ("2022-03-27-08-20-48", "OOD-212"),
    ("2022-03-27-06-19-33", "OOD-211"),
    ("2022-03-27-03-23-34", "OOD-210"),
    ("2022-03-26-23-20-43", "OOD-209"),
    ("2022-03-26-20-31-55", "OOD-208"),
    ("2022-03-25-10-25-37", "OOD-194"),
    ("2022-03-25-08-00-04", "OOD-193"),
    ("2022-03-25-04-51-38", "OOD-192"),
    ("2022-03-25-00-46-17", "OOD-191"),
    ("2022-03-24-21-44-03", "OOD-190"),
    ("2022-03-24-18-02-22", "OOD-189"),
]

TIMESTAMPS_AND_NEPTUNE_RUN_IDS_SHAPES3D = [
    ("2022-03-26-13-41-30", "OOD-207"),
    ("2022-03-26-11-42-36", "OOD-206"),
    ("2022-03-26-09-48-23", "OOD-205"),
    ("2022-03-26-07-54-29", "OOD-204"),
    ("2022-03-26-05-53-30", "OOD-203"),
    ("2022-03-26-04-00-34", "OOD-202"),
    ("2022-03-26-02-02-55", "OOD-201"),
    ("2022-03-26-00-03-21", "OOD-200"),
    ("2022-03-25-22-08-47", "OOD-199"),
    ("2022-03-25-20-22-26", "OOD-198"),
    ("2022-03-25-18-36-38", "OOD-197"),
    ("2022-03-25-16-44-05", "OOD-196"),
]

TIMESTAMPS_AND_NEPTUNE_RUN_IDS = [
    *TIMESTAMPS_AND_NEPTUNE_RUN_IDS_ARROW,
    # *TIMESTAMPS_AND_NEPTUNE_RUN_IDS_SQUARE,
    # *TIMESTAMPS_AND_NEPTUNE_RUN_IDS_DSPRITES,
    # *TIMESTAMPS_AND_NEPTUNE_RUN_IDS_SHAPES3D,
    # ("2022-03-29-19-17-46", "OOD-237"),
]

if __name__ == "__main__":
    for (timestamp_, neptune_run_id_) in TIMESTAMPS_AND_NEPTUNE_RUN_IDS:
        main(timestamp_, neptune_run_id_)
        print("Done!")
