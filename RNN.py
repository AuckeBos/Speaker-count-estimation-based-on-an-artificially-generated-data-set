from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam

from TestSetGenerator import TestSetGenerator
from TimingCallback import TimingCallback
from TrainSetGenerator import TrainSetGenerator
from helpers import write_log

tfd = tfp.distributions
from tensorflow.keras.layers import Dense, InputLayer, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from scipy.stats import poisson


class RNN:
    """
    Recurrent neural network to train speaker count estimation with
    """

    # Training configuration
    batch_size = 64
    num_epochs = 80
    tensorboard_log = f'./tensorboard/{datetime.now().strftime("%m-%d %H:%M")}/'

    # Training callbacks
    callbacks: []

    # If is set, save model to the filename after training
    __save_to_file: str = None

    # The trained network
    __net = None

    # To reproduce
    random_state = 1337

    def __init__(self):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log)
        early_stopping = EarlyStopping(patience=15, verbose=1)
        reduce_lr_on_plateau = ReduceLROnPlateau(factor=.4, patience=7, verbose=1)
        timing = TimingCallback()
        self.callbacks = [tensorboard, early_stopping, reduce_lr_on_plateau, timing]

    def load_from_file(self, file):
        """
        Load the network from filesystem
        :param file: The file path
        """
        self.__net = tf.keras.models.load_model(file)

    def get_net(self, input_shape: tuple):
        """
        Get the network. We use a BiLSTM as described by Stöter et al.
        Input shape: [batch_size, time_steps, n_features]
        :param input_shape the input shape
        :return: The net
        """
        net = Sequential()
        net.add(InputLayer(input_shape=input_shape))
        net.add(Bidirectional(LSTM(30, activation='tanh', return_sequences=True, dropout=0.5)))
        net.add(Bidirectional(LSTM(20, activation='tanh', return_sequences=True, dropout=0.5)))
        net.add(Bidirectional(LSTM(40, activation='tanh', return_sequences=False, dropout=0.5)))

        # net.add(GlobalMaxPool1D())
        net.add(Dense(20, activation='relu'))
        # The network predicts scale parameter \lambda for the poisson distribution
        net.add(Dense(1, activation='exponential'))
        # net.add(tfp.layers.DistributionLambda(tfd.Poisson))

        return net

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
        return tf.keras.losses.MeanSquaredError(y_true, y_hat)
        theta = tf.cast(y_hat, tf.float32)
        y = tf.cast(y_true, tf.float32)
        loss = K.mean(theta - y * K.log(theta + K.epsilon()))
        return loss

    def compile_net(self, input_shape: tuple):
        """
        Get the network and compile and save it
        :param input_shape The input shape
        :return:
        """
        net = self.get_net(input_shape)
        optimizer = Adam(learning_rate=.001)
        net.compile(loss=tf.keras.losses.Poisson(), optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.__net = net
        return self.__net

    # todo
    @staticmethod
    def poisson_loss(y_hat, y_true):
        return -y_hat.log_prob(y_true)

    def __get_train_data(self, files: np.ndarray, min_speakers: int, max_speakers: int, feature_type: str):
        """
        Get train generator and validation set
        - We create a set for validation instead of a generator, such taht we validate on the same set each time
        - This also speeds up validation during the training loop drastically, since we only preprocess the validation set once

        :param files:  All files, will be split .8, 0.2 train,val
        :param min_speakers:  The min number of speakers to generate files for
        :param max_speakers: The max number of speakers to generate files for
        :param feature_type: The feature type
        :return: train_generator, (val_x, val_y)
        """
        # Split files into trai nval
        np.random.shuffle(files)
        split_index = int(len(files) * .8)
        train_files = files[:split_index]
        validation_files = files[split_index:]

        # Train generator: Duplicate all files 5 times
        train_generator = TrainSetGenerator(train_files, self.batch_size, feature_type)
        train_generator.set_limits(min_speakers, max_speakers)
        # train_generator.set_num_files_to_merge(5 * len(train_files))

        # Validation generator: Duplicate all files 2 times
        validation_generator = TrainSetGenerator(validation_files, 1, feature_type)
        validation_generator.set_limits(min_speakers, max_speakers)
        # validation_generator.set_num_files_to_merge(2 * len(validation_files))
        # Generate a full set
        # validation_set = list(validation_generator.__iter__())[0]
        # val_x, val_y = validation_set[0], validation_set[1]


        return train_generator, validation_generator

    def train(self, files: np.ndarray, min_speakers: int, max_speakers: int, feature_type: str):
        """
        Train the network, eg
        - Preprocess the data
        - Train with Adam, negative log likelihood, accuracy metric.
        - Visualize using Tensorboard
        :param files: All files
        :param min_speakers The min number of speakers to generate files for
        :param max_speakers The max number of speakers to generate files for
        :param feature_type:  Feature type to use
        """
        train_generator, validation_generator = self.__get_train_data(files, min_speakers, max_speakers, feature_type)
        net = self.compile_net(train_generator.feature_shape)
        write_log('Training model')
        history = net.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.num_epochs,
            callbacks=self.callbacks,
            verbose=1,
        )
        write_log('Model trained')
        return net, history

    def test(self, X: np.ndarray, Y: np.ndarray, feature_type: str):
        """
        Test the network:
        - Compute the MAE for each count in Y
        - Compute MAE where y in [1, 10]
        - Compute MAE where y in [1, 20]
        - Compute the MAE over all labels
        :param X: The test data set (list of files)
        :param Y: The labels
        :param feature_type: Feature type to use
        :return MAE
        """
        if self.__net is None:
            write_log('Cannot test the network, as it is not initialized. Please train your model, or load it from filesystem', True, True)
        write_log('Testing network')

        generator = TestSetGenerator(X, Y, self.batch_size, feature_type)
        Y_hat = self.__net.predict(generator)

        # Convert predictions to int: take median of poisson distribution
        predictions = np.array([int(poisson(y_hat[0]).median()) for y_hat in Y_hat])
        errors = {}
        for speaker_count in range(min(Y), max(Y) + 1):
            indices_with_count = np.argwhere(Y == speaker_count)
            y_current = Y[indices_with_count]
            predictions_current = predictions[indices_with_count]
            error = mean_absolute_error(y_current, predictions_current)
            average_prediction = np.mean(predictions_current)
            raw_predictions = Y_hat[indices_with_count]
            test = mean_absolute_error([speaker_count] * len(predictions_current), predictions_current)
            errors[speaker_count] = error

        for max_count in [10, 20]:
            indices = np.argwhere(np.logical_and(Y >= 1, Y <= max_count))
            errors[f'1_to_{max_count}'] = mean_absolute_error(Y[indices], predictions[indices])
        errors['mean'] = mean_absolute_error(Y, predictions)
        return errors
