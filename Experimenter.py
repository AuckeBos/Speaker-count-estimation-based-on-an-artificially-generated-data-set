import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader
from RNN import RNN
from TrainSetGenerator import TrainSetGenerator


class Experimenter:
    train_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TRAIN'
    test_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TEST'
    libri_dir = './data/LibriCount/test'

    dest_dir = './data/experiments'

    feature_options = TrainSetGenerator.FEATURE_OPTIONS

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
        experiments = {}
        train_data = self.__get_train_data()
        test_data = self.__get_test_data()
        for train_data_current in train_data:
            experiment_for_trainset = {}
            files = train_data_current['files']
            min_speakers = train_data_current['min_speakers']
            max_speakers = train_data_current['max_speakers']
            for feature_type in self.feature_options:
                experiment_for_feature = {}
                name = f'./trained_networks_with_generators/rnn_train_{min_speakers}_{max_speakers}/{feature_type}'
                network, history = self.__train_net(files, min_speakers, max_speakers, feature_type, name)
                history = history.history
                history['lr'] = [float(lr) for lr in history['lr']]
                experiment_for_feature['history'] = history

                # Now load best performing model
                network.load_from_file(name)

                # Test performance
                for test_name, test_data_current in test_data.items():
                    x, y = test_data_current['x'], test_data_current['y']
                    experiment_for_feature[test_name] = self.__test_net(network, x, y, feature_type)

                experiment_for_trainset[feature_type] = experiment_for_feature
            experiments[f'train_{min_speakers}_{max_speakers}'] = experiment_for_trainset
        with open('experiments_with_generator.json', 'w+') as fp:
            json.dump(experiments, fp)

    def __train_net(self, files: np.ndarray, min_speakers: int, max_speakers: int, feature_type: str, save_to: str):
        """
        Train a network
        :param files: The train files
        :param min_speakers: The min number of speakers to generate files for
        :param max_speakers: The max number of speakers to generate files for
        :param feature_type: The feature type to use
        :param save_to:  Location to save the best performing model to
        :return: RNN, history
        """
        network = RNN()
        network.save_to_file(save_to)
        _, history = network.train(files, min_speakers, max_speakers, feature_type)
        return network, history

    def __test_net(self, network: RNN, x: np.ndarray, y: np.ndarray, feature_type: str):
        """
        Test a trained network
        :param network:  The trained network
        :param x: The test files (pre-merged)
        :param y:  The corresponding labels
        :param feature_type:  The feature type, must be the same as used for traning
        :return: MAE on different levels
        """
        return network.test(x, y, feature_type)

    def __get_train_data(self):
        """
        The train data is a list of dicts. Each dict defines a training data configuration.  Each configuration defines:
        - min_speakers: The min speaker count for the data generator
        - max_speakers: The max speaker count for the data generator
        - files: The files the generator will use
        :return: The list
        """
        files = glob.glob(f'{self.train_dir}/*.WAV')
        return [
            {
                'min_speakers': 1,
                'max_speakers': 10,
                'files': files,
            },
            {
                'min_speakers': 1,
                'max_speakers': 20,
                'files': files,
            }
        ]

    def __get_test_data(self):
        """
        The test data are actually X, Y, where X is a list of filenames with pre-merged wavs.
        We pre-merge them, such that we have the same test set each time
        :return: {
            'test_set_type' : {
                'x': pre-merged wav files
                'y': labels
            }
        }
        """
        # Load libri
        libri_x, libri_y = self.__load_libri()
        data = {
            'libri': {
                'x': libri_x,
                'y': libri_y
            }
        }
        # Load two versions of timit
        for (min_speakes, max_speakers) in [(1, 10), (1, 20)]:
            # If not yet exist, generate data. Then save test x and y in np arrays
            test_x, test_y = self.__load_timit_test(min_speakes, max_speakers)
            data[f'{min_speakes}_to_{max_speakers}'] = {
                'x': test_x,
                'y': test_y
            }
        return data

    def visualize(self, file: str):
        """
        Plot results of run()
        :param file: experiments.json
        """
        # Load results
        content = None
        with open(file) as json_file:
            content = json.load(json_file)
        # fig, axs = plt.subplots(2)
        for fig_i, train_set in enumerate(['train_1_10', 'train_1_20']):
            plt.figure(fig_i)
            # axs[fig_i].title(train_set)
            data = content[train_set]
            for feature in self.feature_options:
                color = np.random.rand(3, )
                feature_data = data[feature]
                x, y = [], []
                for i in range(1, 21):
                    if str(i) in feature_data['1_to_10']:
                        x.append(i)
                        y.append(feature_data['1_to_10'][str(i)])
                # mae_mean = feature_data['1_to_20']['1_to_10']
                plt.plot(x, y, label=feature, c=color)
                # plt.plot(mae_mean, 'o', c=color)
            plt.title(train_set)
            plt.ylabel('MAE')
            plt.xlabel('Max number of speakers')
            plt.legend(loc='upper right')
            plt.ylim(0, 10)

        # plt.title('MAE by feature representation, trained on 1-10')
        plt.ylabel('MAE')
        plt.ylim(0, 10)
        plt.xlabel('Max number of speakers')
        plt.legend(loc='upper right')
        plt.show()

    def test_networks(self):
        """
        Test the networks saved by run()
        """
        test_x_max_10, test_y_max_10 = self.__load_timit_test(1, 10)
        test_x_max_20, test_y_max_20 = self.__load_timit_test(1, 20)
        libri_x, libri_y = self.__load_libri()
        for train_set in ['1_10', '1_20']:
            for feature_type in self.feature_options:
                name = f'./trained_networks_with_generators/rnn_{train_set}/{feature_type}'
                network = RNN()
                network.load_from_file(name)
                errors_max_10 = network.test(test_x_max_10, test_y_max_10, feature_type)
                errors_max_20 = network.test(test_x_max_20, test_y_max_20, feature_type)
                errors_max_libri = network.test(libri_x, libri_y, feature_type)

    def __load_timit_test(self, min_count, max_count):
        """
        Load timit test set
        :param min_count: The minimum max number of speakers per file
        :param max_count:  The maximum max number of speakers per file
        :return:  test_x, test_y
        """
        test_dest_dir = f"{self.dest_dir}/{min_count}_to_{max_count}/test"
        data_loader = DataLoader(self.train_dir, self.test_dir, test_dest_dir)
        data_loader.min_speakers = min_count
        data_loader.max_speakers = max_count
        _, (test_x, test_y) = data_loader.load_data()
        return test_x, test_y

    def __load_libri(self):
        """
        Load the LibriCount dataset
        :return: libri_x, libri_y
        """
        return DataLoader.load_libricount(self.libri_dir)
