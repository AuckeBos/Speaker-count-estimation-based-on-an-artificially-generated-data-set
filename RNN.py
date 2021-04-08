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

    # Create 40 mel filters
    num_mel_filters = 40

    # If using MFCC, take coefficients 1-13
    num_coefficients = 12

    # Use a frame length of 25ms
    frame_length = int(sample_rate * 0.025)

    # Use overlap of 10ms
    n_overlap = int(sample_rate * 0.01)

    # 501 * num_mel_filters
    input_size = (int(sample_rate * seconds_per_record / n_overlap) + 1, num_mel_filters)

    batch_size = 64
    num_epochs = 60
    tensorboard_log = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}/'

    # Training callbacks
    callbacks: []

    # If is set, save model to the filename after training
    __save_to_file: str = None

    # The trained network
    __net = None

    # If true, use MFCC 1-12. If false, use STFT
    __use_mfcc = False

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
        net.add(Bidirectional(LSTM(40, activation='tanh', return_sequences=True, dropout=0.5)))

        net.add(GlobalMaxPool1D())
        net.add(Dense(20, activation='relu'))
        # The network predicts scale parameter \lambda for the poisson distribution
        net.add(Dense(1, activation='exponential'))
        print(net.summary())
        return net

    def use_mfcc(self):
        """
        Set use_mfcc = true. Also update self.input_size to match new size (since we now have num_coefficients features per frame, instead of num_filters)
        """
        self.__use_mfcc = True
        self.input_size = (self.input_size[0], self.num_coefficients)

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
        train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
        validation_iter = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(self.batch_size)

        write_log('Training model')
        history = net.fit(
            train_iter,
            validation_data=validation_iter,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=self.callbacks,
        )
        write_log('Model trained')

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
        features = np.array([librosa.feature.melspectrogram(S=x, sr=self.sample_rate, n_fft=self.frame_length, hop_length=self.n_overlap, n_mels=self.num_mel_filters) for x in stft])
        if self.__use_mfcc:
            features = np.array([librosa.feature.mfcc(S=x, sr=self.sample_rate, n_mfcc=self.num_coefficients + 1)[1:] for x in features])

        # Reshape to [batch_size, time_steps, n_features]
        features = features.reshape((features.shape[0], features.shape[2], features.shape[1]))
        write_log('Data preprocessed')
        return features

    def test(self, X, Y):
        """
        Test the network, and print the predictions to stdout
        :param X: The test data set
        :param Y: The labels
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
