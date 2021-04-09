import json

import numpy as np

from DataLoader import DataLoader
from RNN import RNN


class Experimenter:
    train_src_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TRAIN'
    test_src_dr = './data/TIMITS/TIMIT/wavfiles16kHz/TEST'

    libri_dir = './data/LibriCount/test'

    dest_dir = './data/experiments'

    feature_options = RNN.FEATURE_OPTIONS

    def run(self):
        """
        Run the experiment:
        - For two training sets, one containing max speakers of 1-10, one containing 1-20
            - For 5 different features types
                - Train the network
                - Load the best performing version into the network
                - Test the network
                - Save training history and testing performance to json
        - Save results to json
        :return:
        """
        # Todo: use multithreader
        # https://stackoverflow.com/questions/56344611/how-can-take-advantage-of-multiprocessing-and-multithreading-in-deep-learning-us
        experiments = {}
        train_x_max_10, train_y_max_10, test_x_max_10, test_y_max_10 = self.__load_timit(1, 10)
        train_x_max_20, train_y_max_20, test_x_max_20, test_y_max_20 = self.__load_timit(1, 20)
        libri_x, libri_y = self.__load_libri()

        train_sets = {
            'train_max_10': (train_x_max_10, train_y_max_10),
            'train_max_20': (train_x_max_20, train_y_max_20)
        }
        for index, (train_x, train_y) in train_sets.items():
            experiment_for_trainset = {}
            for feature_type in self.feature_options:
                experiment_for_feature = {}
                name = f'./trained_networks/rnn_{index}/{feature_type}'
                network, history = self.__train_net(train_x, train_y, feature_type, name)
                history = history.history
                history['lr'] = [float(lr) for lr in history['lr']]
                experiment_for_feature['history'] = history

                # Now load best performing model
                network.load_from_file(name)

                # Test performance
                experiment_for_feature['1_to_10'] = self.__test_net(network, test_x_max_10, test_y_max_10)
                experiment_for_feature['1_to_20'] = self.__test_net(network, test_x_max_20, test_y_max_20)
                experiment_for_feature['libri'] = self.__test_net(network, libri_x, libri_y)
                experiment_for_trainset[feature_type] = experiment_for_feature
            experiments[index] = experiment_for_trainset
        with open('experiments.json', 'w+') as fp:
            json.dump(experiments, fp)

    def __test_net(self, network: RNN, x, y):
        """
        Test a trained network
        :param network:  The trained network
        :param x:
        :param y:
        :return: The test results: Dictionary with MAE on different levels
        """
        return network.test(x, y)

    def __train_net(self, x, y, feature_type, save_to):
        """
        Train a network
        :param x:
        :param y:
        :param feature_type: Which feature type the model should use
        :param save_to: To which location the best performing model should be saved
        :return: The trained model (instance of RNN), and its history
        """
        network = RNN()
        network.save_to_file(save_to)
        network.set_feature_type(feature_type)
        _, history = network.train(x, y)
        return network, history

    def __load_timit(self, min_count, max_count):
        """
        Load timit dataset
        :param min_count: The minimum max number of speakers per file
        :param max_count:  The maximum max number of speakers per file
        :return:  train_x, train_y, test_x, test_y
        """
        train_dir = f"{self.dest_dir}/{min_count}_to_{max_count}/train"
        test_dir = f"{self.dest_dir}/{min_count}_to_{max_count}/test"
        data_loader = DataLoader(self.train_src_dir, self.test_src_dr, train_dir, test_dir)
        data_loader.min_speakers = min_count
        data_loader.max_speakers = max_count
        return data_loader.load_data()

    def __load_libri(self):
        """
        Load the LibriCount dataset
        :return: libri_x, libri_y
        """
        return DataLoader.load_libricount(self.libri_dir)
