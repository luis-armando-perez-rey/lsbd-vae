import numpy as np
import tensorflow as tf


def print_and_log_time(start, end, neptune_run=None, neptune_log_name=None):
    elapsed_time = end - start
    minutes, seconds = np.divmod(elapsed_time, 60)
    hours, minutes = np.divmod(minutes, 60)
    print("Time elapsed:", hours, "hours,", minutes, "minutes, and", seconds, "seconds")
    if neptune_run is not None and neptune_log_name is not None:
        neptune_run[neptune_log_name + "/total_seconds"] = elapsed_time
        neptune_run[neptune_log_name + "/hours"] = hours
        neptune_run[neptune_log_name + "/minutes"] = minutes
        neptune_run[neptune_log_name + "/seconds"] = seconds


def file_exists(filepath, extension=None, overwrite=False):
    if overwrite:
        return False
    else:
        if extension is not None:
            filepath = filepath.parent / (filepath.name + extension)
        if filepath.is_file():
            print(f"File {filepath} already exists, skipping. Set overwrite=True to overwrite.")
        return filepath.is_file()


class NeptuneMonitor(tf.keras.callbacks.Callback):
    def __init__(self, neptune_run):
        super().__init__()
        self.neptune_run = neptune_run

    def on_epoch_end(self, epoch, logs=None):
        for metric_name, metric_values in logs.items():
            self.neptune_run["metrics/" + metric_name].log(metric_values)
