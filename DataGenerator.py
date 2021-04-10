import librosa
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.utils import Sequence
from scipy.io import wavfile

from helpers import write_log

tfd = tfp.distributions


class DataGenerator(Sequence):
    """
    DataGenerator, used to train, validate, and test the network
    """
    # 5 seconds per file, at 16KhZ
    seconds_per_record = 5
    sample_rate = 16000

    # For now, pad to and cut off at 5 seconds
    pad_to = sample_rate * 5

    # If feature_type is MELXX, num_mel_filters will be set to XX in set_feature_type()
    num_mel_filters: int = None

    # If using MFCC, take coefficients 1-13
    num_coefficients = 12

    # Use a frame length of 25ms
    frame_length = int(sample_rate * 0.025)

    # Use overlap of 10ms
    n_overlap = int(sample_rate * 0.01)

    # list of filenames
    x: np.ndarray
    # list of speaker counts
    y: np.ndarray

    # Batch size
    batch_size: int

    # If true, shuffle on epoch end
    shuffle: bool

    # Use values in index to get indices
    indices: np.ndarray

    # For reproduction
    random_state = 1337

    # Valid features types are provided below
    # The feature type defines how we preprocess our data, and also defines the input shape
    feature_type: str = None

    FEATURE_TYPE_STFT = 'STFT'
    FEATURE_TYPE_LOG_STFT = 'LOG_STFT'
    FEATURE_TYPE_MEL_20 = 'MEL20'
    FEATURE_TYPE_MEL_40 = 'MEL40'
    FEATURE_TYPE_MFCC = 'MFCC'

    FEATURE_OPTIONS = [
        FEATURE_TYPE_STFT,
        FEATURE_TYPE_LOG_STFT,
        FEATURE_TYPE_MEL_20,
        FEATURE_TYPE_MEL_40,
        FEATURE_TYPE_MFCC,
    ]

    # Feature shape is set when feature_type is set. Used in network to define input shape
    feature_shape: tuple

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, feature_type: str, shuffle: bool = True):
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
        self.shuffle = shuffle
        self.set_feature_type(feature_type)
        self.on_epoch_end()

    def set_feature_type(self, type: str):
        """
        Set on init.
        Define the feature representation
        - MEL with 20 or 40 filters
        - OR MFCC with coefficients 1-13
        - OR STFT, either logarithmically scaled or normal transforms
        - Also set self.feature_shape depending on the  feature type
        :param type: The feature type
        """
        # STFT in mel basis, using 20 filters
        if type == self.FEATURE_TYPE_MEL_20:
            self.num_mel_filters = 20
            # 501 * 20
            shape = self.num_mel_filters
        # STFT in mel basis, using 20 filters
        elif type == self.FEATURE_TYPE_MEL_40:
            self.num_mel_filters = 40
            # 501 * 40
            shape = self.num_mel_filters
        # Mel frequency cepstral coefficients 1-13
        elif type == self.FEATURE_TYPE_MFCC:
            # 501 * 12
            shape = self.num_coefficients
        # Short time fourier transform: windows in frequency domains, 501 * (1 + self.frame_length / 2) = 501 * 201
        elif type == self.FEATURE_TYPE_STFT or type == self.FEATURE_TYPE_LOG_STFT:
            # 501 * 201
            shape = 1 + self.frame_length // 2
        else:
            write_log(f'Feature type {type} is invalid', True, True)
        write_log(f'Using feature type {type}')
        self.feature_type = type
        self.feature_shape = (int(self.sample_rate * self.seconds_per_record / self.n_overlap) + 1, shape)

    @staticmethod
    def get_shape_for_type(feature_type: str):
        """
        Static method to retrieve the input shape for a certain feature type
        Stubs a generator with fake data, returns its input shape
        :param feature_type:
        :return:
        """
        generator = DataGenerator(np.array([]), np.array([]), 0, feature_type)
        return generator.feature_shape

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        """
        On epoch end, shuffle indices if desired
        """
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indices)

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
        x = self.__preprocess(wavs)
        y = labels

        return np.array(x), np.array(y)

    def __use_stft(self):
        """
        True if self.__feature_type is either STFT or LOG_STFT
        :return: bool
        """
        return 'STFT' in self.feature_type

    def __use_log_stft(self):
        """
        True if self.__feature_type is LOG_STFT
        :return: bool
        """
        return self.feature_type == self.FEATURE_TYPE_LOG_STFT

    def __use_mfcc(self):
        """
        True if self.__feature_type is MFCC
        :return: bool
        """
        return self.feature_type == self.FEATURE_TYPE_MFCC

    def __preprocess(self, X):
        """
        Preprocess X
        :param X:
        :return: prepocessed X
        """
        # Read wav files
        X = np.array([record for (_, record) in [wavfile.read(wav) for wav in X]], dtype='object')
        # Merge all dimensions. For speaker_count > 1, each speaker initially has its own dimension
        X = np.array([np.sum(x, axis=1) if x.ndim > 1 else x for x in X], dtype='object')
        # Pad to and cut off at 5 seconds
        # todo: improve htis
        X = np.array([np.pad(x, (0, max(self.pad_to - len(x), 0)))[:self.pad_to] for x in X])

        # Load Short Time Fourier Transformas
        stft = np.abs([librosa.stft(np.array(x, dtype=float), n_fft=self.frame_length, hop_length=self.n_overlap) for x in X])
        stft = librosa.util.normalize(stft)
        if self.__use_mfcc():
            # MFCC
            log_spectogram = [librosa.power_to_db(x) for x in stft]
            features = np.array([librosa.feature.mfcc(S=x, sr=self.sample_rate, n_mfcc=self.num_coefficients + 1)[1:] for x in log_spectogram])
        else:
            if self.__use_stft():
                # STFT or LOG_STFT
                if self.__use_log_stft():
                    # LOG_STFT
                    features = np.log(stft + tf.keras.backend.epsilon())
                else:
                    # STFT
                    features = stft
            else:
                # MEL20/MEL40
                features = np.array([librosa.feature.melspectrogram(S=x, sr=self.sample_rate, n_fft=self.frame_length, hop_length=self.n_overlap, n_mels=self.num_mel_filters) for x in stft])

        # Reshape to [batch_size, time_steps, n_features]
        features = features.reshape((features.shape[0], features.shape[2], features.shape[1]))
        return features
