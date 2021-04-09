from datetime import datetime

import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam

from helpers import write_log

tfd = tfp.distributions
from tensorflow.keras.layers import Dense, InputLayer, Bidirectional, LSTM, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from scipy.stats import poisson


class RNN:
    """
    Recurrent neural network to train speaker count estimation with
    """
    # 5 seconds per file, at 16KhZ
    seconds_per_record = 5
    sample_rate = 16000

    # If feature_type is MELXX, num_mel_filters will be set to XX in set_feature_type()
    num_mel_filters: int = None

    # If using MFCC, take coefficients 1-13
    num_coefficients = 12

    # Use a frame length of 25ms
    frame_length = int(sample_rate * 0.025)

    # Use overlap of 10ms
    n_overlap = int(sample_rate * 0.01)

    # Input size will be set depending on the feature_types, this value is set in set_feature_type.
    # The first value depends on the number of timeframes, this is equal for all feature types: 501
    # The second value is the number of features per window, it is set in set_feature_type
    input_size: tuple = (int(sample_rate * seconds_per_record / n_overlap) + 1, None)

    # Training configuration
    batch_size = 64
    num_epochs = 60
    tensorboard_log = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}/'

    # Training callbacks
    callbacks: []

    # If is set, save model to the filename after training
    __save_to_file: str = None

    # The trained network
    __net = None

    # Valid features types are provided below
    # The feature type defines how we preprocess our data, and also defines the input shape
    __feature_type: str = None

    FEATURE_TYPE_STFT = 'STFT'
    FEATURE_TYPE_LOG_STFT = 'LOG_STFT'
    FEATURE_TYPE_MEL_20 = 'MEL20'
    FEATURE_TYPE_MEL_40 = 'MEL40'
    FEATURE_TYPE_MFCC = 'MFCC'

    # To reproduce
    random_state = 1000

    def __init__(self):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log)
        early_stopping = es = EarlyStopping(patience=10, verbose=1)
        reduce_lr_on_plateau = ReduceLROnPlateau(factor=.4, patience=4, verbose=1)
        self.callbacks = [tensorboard, early_stopping, reduce_lr_on_plateau]

    def load_from_file(self, file):
        """
        Load the network from filesystem
        :param file: The file path
        """
        self.__net = tf.keras.models.load_model(file)

    def get_net(self):
        """
        Get the network. We use a BiLSTM as described by Stöter et al.
        Input shape: [batch_size, time_steps, n_features]
        :return: The net
        """
        net = Sequential()
        net.add(InputLayer(input_shape=self.input_size))
        net.add(Bidirectional(LSTM(30, activation='tanh', return_sequences=True, dropout=0.5)))
        net.add(Bidirectional(LSTM(20, activation='tanh', return_sequences=True, dropout=0.5)))
        net.add(Bidirectional(LSTM(40, activation='tanh', return_sequences=False, dropout=0.5)))

        # net.add(GlobalMaxPool1D())
        net.add(Dense(20, activation='relu'))
        # The network predicts scale parameter \lambda for the poisson distribution
        net.add(Dense(1, activation='exponential'))
        print(net.summary())
        return net

    def set_feature_type(self, type: str):
        """
        Define the feature representation
        - MEL with 20 or 40 filters
        - OR MFCC with coefficients 1-13
        - OR STFT, either logarithmically scaled or normal transforms
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
        self.__feature_type = type
        self.input_size = (self.input_size[0], shape)

    def save_to_file(self, file):
        """
        Set the filename to save our best performing model to
        Also add callback to save the best model
        :param file:
        """
        self.__save_to_file = file
        self.callbacks.append(ModelCheckpoint(file, save_best_only=True))

    @staticmethod
    def poisson(y_true, y_hat):
        """
        Since we are predicting a Poisson distribution, our loss function is the poisson loss
        :param y_true: Number of speakers
        :param y_hat: Lambda for poisson
        :return:
        """
        theta = tf.cast(y_hat, tf.float32)
        y = tf.cast(y_true, tf.float32)
        loss = K.mean(theta - y * K.log(theta + K.epsilon()))
        return loss

    def compile_net(self):
        """
        Get the network and compile and save it
        :return:
        """
        net = self.get_net()
        optimizer = Adam(learning_rate=.001)
        net.compile(loss=self.poisson, optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.__net = net
        return self.__net

    def train(self, x, y):
        """
        Train the network, eg
        - Preprocess the data
        - Train with Adam, negative log likelihood, accuracy metric.
        - Visualize using Tensorboard
        :param x:
        :param y:
        """
        x = self.__preprocess(x)
        y = np.array(y).astype(int)

        net = self.compile_net()

        # Split and create DataSets
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=.2, random_state=self.random_state)

        write_log('Training model')
        history = net.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            callbacks=self.callbacks,
            verbose=1,
        )
        write_log('Model trained')

    def __use_stft(self):
        """
        True if self.__feature_type is either STFT or LOG_STFT
        :return: bool
        """
        return 'STFT' in self.__feature_type

    def __use_log_stft(self):
        """
        True if self.__feature_type is LOG_STFT
        :return: bool
        """
        return self.__feature_type == self.FEATURE_TYPE_LOG_STFT

    def __use_mfcc(self):
        """
        True if self.__feature_type is MFCC
        :return: bool
        """
        return self.__feature_type == self.FEATURE_TYPE_MFCC

    def __preprocess(self, X):
        """
        Preprocess X, as described by Stöter et al in CountNet
        - Generate Short Time Fourier Transforms with frames of 25 ms, 10 ms overlap
        - From the STFT, generate 40 mel filters
        - Apply log
        :param X:
        :return:
        """
        write_log('Preprocessing data')
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
        write_log('Data preprocessed')
        return features

    def test(self, X, Y):
        """
        Test the network, and print the predictions to stdout
        :param X: The test data set
        :param Y: The labels
        :return MAE
        """
        if self.__net is None:
            write_log('Cannot test the network, as it is not initialized. Please train your model, or load it from filesystem', True, True)
        write_log('Testing network')
        X = self.__preprocess(X)
        Y_hat = self.__net.predict(X)
        # Convert predictions to int: take median of poisson distribution
        predictions = [int(poisson(y_hat).median()) for y_hat in Y_hat]
        for (y_hat, y) in zip(predictions, Y):
            print(f'Predicted {y_hat:02d} for {y:02d} (difference of {abs(y_hat - y)})')
        error = mean_absolute_error(Y, predictions)
        write_log(f'MAE: {error}')
        return error
