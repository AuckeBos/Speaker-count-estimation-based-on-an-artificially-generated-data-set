import math

import numpy as np
import tensorflow_probability as tfp

from TrainSetGenerator import TrainSetGenerator
from scipy.io import wavfile

tfd = tfp.distributions


class TestSetGenerator(TrainSetGenerator):
    """
    The TestSetGenerator consumes pre-merged files. Provided X is a list of merged files, Y a corresponding list of labels.
    Extends TrainSetGenerator, since it used many of its functionality
    The main difference is the generation of datapoints: We have pre-merged files, and thus do not need to merge them ourselves
    """
    # list of filenames
    x: np.ndarray
    # list of speaker counts
    y: np.ndarray

    # Use values in index to get indices
    indices: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, feature_type: str):
        """
        Initialize Generator
        :param x: List of filenames
        :param y: Corresponding list of speaker counts
        :param batch_size:  Batch size
        :param feature_type:  Type of features to use. See set_feature_type
        :param shuffle:  If true, shuffle indices on epoch end
        """
        self.indices = np.arange(len(x))
        self.x = np.array(x)
        self.y = np.array(y)

        self.batch_size = batch_size
        self._set_feature_type(feature_type)
        # Disable augmentation
        self.augment = False

    def on_epoch_end(self):
        """
        No on epoch end
        """
        pass

    def __len__(self):
        """
        Use ceil to support the last, possible smaller, batch
        :return:
        """
        return int(math.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, batch_index):
        """
        Get a batch
        :param batch_index: The batch index
        :return: x,y: ndarrays
        """
        indices = self.indices[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        wavs = self.x[indices]
        labels = self.y[indices]

        # read your data here using the batch lists, batch_x and batch_y
        x = self._preprocess(wavs)
        y = labels

        return np.array(x), np.array(y)

    def _preprocess(self, X):
        """
        Preprocess X
        :param X: List of filenames
        :return: prepocessed X
        """
        # Read wav files
        X = np.array([record for (_, record) in [wavfile.read(wav) for wav in X]], dtype='object')
        # Now use parent preprocess, which assumes loaded files
        return super()._preprocess(X)
