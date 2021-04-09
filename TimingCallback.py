from timeit import default_timer as timer
import tensorflow as tf


class TimingCallback(tf.keras.callbacks.Callback):
    starttime: float = None

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        logs['timer'] = timer() - self.starttime
