import math

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.io import wavfile
from tensorflow.keras.utils import Sequence

from helpers import write_log

tfd = tfp.distributions


class TrainSetGenerator(Sequence):
    """
    The TrainSetGenerator generates batches for training and validation sets.

    - As input, we get only a list of unmerged files.
    - We merge these files on the fly. The number of files that is merged defines the label, and is alway between self.min_speakers and self.max_speakers
    - Since we merge randomly, we allow to duplicate the input list of files. This gives us more training data, and this data will most likely not contain duplicate files
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

    # list of available files (filenames)
    files: np.ndarray

    # The number of files to merge. Can be > len(files), since we randomly select files from the list each time
    # If not provided, set to int(mean(range(min_speakers, max_speakers)) * len(files)). Such that we generate
    # ~len(files) files
    num_files_to_merge: int

    # Batch size
    batch_size: int

    # If true, shuffle on epoch end
    shuffle: bool

    # Labels are generated on epoch end
    labels: np.ndarray

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

    # Create wav files of min_speakers - max_speakers concurrent speakers
    min_speakers = 1
    max_speakers = 20

    def __init__(self, files: np.ndarray, batch_size: int, feature_type: str, shuffle: bool = True):
        """
        Initialize Generator
        :param files: List of filenames
        :param batch_size:  Batch size. The final batch may be smaller
        :param feature_type:  Type of features to use. See set_feature_type
        :param shuffle:  If true, shuffle indices on epoch end
        """
        self.files = np.array(files)
        self.num_files_to_merge = int(np.mean(range(self.min_speakers, self.max_speakers)) * len(self.files))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self._set_feature_type(feature_type)
        self.__generate_labels()
        self.on_epoch_end()

    def set_num_files_to_merge(self, amount: int):
        """
        Set the number of files to merge. Can be >len(files), since we randomly select files from the list on the fly,
        so files can be used more than once
        :param amount: The total amount of files to merge
        """
        self.num_files_to_merge = amount
        self.__generate_labels()
        self.on_epoch_end()

    def set_limits(self, min_speaker_count: int, max_speaker_count: int):
        """
        Set the limits for the labels to be generated. Also (re-)generate the labels for these new limits
        :param min_speaker_count: Minimal speakers per file
        :param max_speaker_count:  Maximal speakers per file
        """
        self.min_speakers = min_speaker_count
        self.max_speakers = max_speaker_count
        self.num_files_to_merge = int(np.mean(range(self.min_speakers, self.max_speakers)) * len(self.files))
        self.__generate_labels()
        self.on_epoch_end()

    def __generate_labels(self):
        """
        Define which files will be merged.
        - Evenly distribute the labels between self.min_speakers and self.max_speakers
        - The sum of the labels is equal to self.num_files_to_merge
        :return:
        """
        # Labels max to min
        available_labels = range(self.max_speakers, self.min_speakers - 1, -1)
        labels = []
        while (sum(labels) != self.num_files_to_merge):
            for label in available_labels:
                if (sum(labels) + label > self.num_files_to_merge):
                    continue
                labels.append(label)
        # Taken contains evenly distributed labels between [min, max], the sum is exactly equal to the number of available files
        self.labels = np.array(labels)

    @staticmethod
    def get_shape_for_type(feature_type: str):
        """
        Static method to retrieve the input shape for a certain feature type
        Stubs a generator with fake data, returns its input shape
        :param feature_type:
        :return:
        """
        generator = TrainSetGenerator(np.array(['']), 0, feature_type)
        return generator.feature_shape

    def on_epoch_end(self):
        """
        On epoch end, shuffle indices if desired
        """
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.files)
            np.random.seed(self.random_state)
            np.random.shuffle(self.labels)

    def _set_feature_type(self, type: str):
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

    def __len__(self):
        return int(math.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, batch_index):
        """
        Get a batch
        - Get the labels for the batch
        - Create datapoints for these labels, by poping files from the remaining files list and merging them
        :param batch_index: The batch index
        :return: x,y: ndarrays
        """
        labels = self.labels[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        y = labels
        x = [self.__get_datapoint(speaker_count) for speaker_count in y]

        # Now preprocess the files
        x = self._preprocess(x)

        return np.array(x), np.array(y)

    def __get_datapoint(self, speaker_count: int):
        """
        Get one datapoint:
        - Load get the first speaker_count number of files
        - Remove those failes from the remaining files
        - Load the wav data of those files
        :param speaker_count: The number of files to merge
        :return: List containing wav data
        """
        files = np.random.choice(self.files, speaker_count, replace=False)
        datapoint = self.__merge_files(files)
        return datapoint

    def __merge_files(self, files):
        """
        Merge an amount of files into one numpy array
        :param files:
        :return: List containing wav data
        """
        # Load ata
        data = [record for (_, record) in [wavfile.read(wav) for wav in files]]
        # Pad to longest file
        pad_to = len(max(data, key=len))
        data = np.array([np.pad(x, (0, pad_to - len(x))) for x in data])
        # Sum over files
        data = np.sum(data, axis=0, dtype=float)
        return data

    def __randomize_loudness(self, wav):
        # Randomize to between 50-70 DB
        normalize_to = np.random.uniform(50, 70)
        # Calculate current loudness
        # First normalize, to make sure we taking a valid log
        wav = wav / np.max(np.abs(wav))
        loudness = 10 * math.log10(np.sum(wav ** 2) / float(len(wav) * 4 * 10 ** -10))
        # Calc multiplier to use to get to the loudness of normalize_to
        muliplier = 10 ** ((normalize_to - loudness) / float(20))
        # Transform into the right loudness
        wav = wav * muliplier
        return wav

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

    def _preprocess(self, X):
        """
        Preprocess X
        :param X: (Batch) Numpy array containing wav data
        :return: prepocessed X
        """
        # Merge all dimensions. For speaker_count > 1, each speaker initially has its own dimension (for test files)
        X = np.array([np.sum(x, axis=1) if x.ndim > 1 else x for x in X], dtype='object')

        # Randomize loudness to 50-70 db for each file
        X = np.array([self.__randomize_loudness(x) for x in X], dtype='object')

        # Normalize to -1, 1
        X = [x / np.max(np.abs(x)) for x in X]
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
