from datetime import datetime

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow_probability as tfp
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import Normalizer

from helpers import write_log

tfd = tfp.distributions
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, InputLayer, Bidirectional, LSTM, MaxPool1D, GlobalMaxPool1D
from tensorflow.keras.models import Sequential


class RNN:
    """
    Recurrent neural network to train speaker count estimation with
    """
    # 5 seconds per file, at 16KhZ
    seconds_per_record = 5
    sample_rate = 16000
    # Create 40 mel filters
    num_mel_filters = 40

    # Use a frame length of 25ms
    frame_length = int(sample_rate * 0.025)

    # Use overlap of 10ms
    n_overlap = int(sample_rate * 0.01)

    # 501 * 40
    input_size = (int(sample_rate * seconds_per_record / n_overlap) + 1, num_mel_filters)

    batch_size = 8
    num_epochs = 50
    tensorboard_log = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}/'

    def get_net(self):
        """
        Get the network. We use a BiLSTM as described by Stöter et al.
        :return:
        """
        net = Sequential()
        net.add(InputLayer(input_shape=self.input_size, batch_size=self.batch_size))
        net.add(Bidirectional(LSTM(30, activation='relu', return_sequences=True)))
        net.add(Bidirectional(LSTM(20, activation='relu', return_sequences=True)))
        net.add(Bidirectional(LSTM(40, activation='relu', return_sequences=True)))
        net.add(GlobalMaxPool1D())
        net.add(Dense(40, activation='relu'))
        net.add(Dense(1, activation='exponential'))
        # The network predicts scale parameter \lambda for the poisson distribution
        net.add(tfp.layers.DistributionLambda(tfp.distributions.Poisson))
        return net

    @staticmethod
    def loss(y_true, y_hat):
        """
        Since we are predicting a Poisson distribution, our loss function is the negative log likelihood function
        :param y_true:
        :param y_hat:
        :return:
        """
        return -y_hat.log_prob(y_true)

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
        y = np.array(y, dtype=float)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
        train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
        test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log)

        net = self.get_net()
        optimizer = Adam(learning_rate=.001)
        net.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])
        write_log('Training model')
        history = net.fit(
            train_iter,
            validation_data=test_iter,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=[tensorboard_callback],
        )
        write_log('Model trained')
        pass

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
        mel = [librosa.feature.melspectrogram(S=x, sr=self.sample_rate, n_fft=self.frame_length, hop_length=self.n_overlap, n_mels=self.num_mel_filters) for x in stft]
        mel = np.log(mel)
        write_log('Data preprocessed')
        return mel
