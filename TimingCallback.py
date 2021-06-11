from timeit import default_timer as timer
import tensorflow as tf


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Add epoch runtimes to logs
    """
    starttime: float = None

    def on_epoch_begin(self, epoch, logs=None):
        """
        Start the timer on the start of each epoch
        :param epoch:
        :param logs:
        """
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        """
        Log the time of the past epoch
        :param epoch:
        :param logs:
        :return:
        """
        logs['timer'] = timer() - self.starttime
