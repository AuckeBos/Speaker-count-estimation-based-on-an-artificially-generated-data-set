import glob
import json
import csv

import matplotlib.pyplot as plt
import numpy as np

from DataLoader import DataLoader
from RNN import RNN
from TrainSetGenerator import TrainSetGenerator
from scipy import stats
from matplotlib import rc


def flatten(S):
    """
    Helper function to recursively flatten a list
    :param S:  The nested list
    :return:  The flattened list
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


class Experimenter:
    """
    Provides functions to run the experiments as described in Section 4 of the paper.
    """

    # Data directories
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
        with open('experiments.json', 'w+') as fp:
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
        We pre-merge them, such that we have the same test set each time. Are created and saved to FS using the DataLoader
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

    def visualize_newset_generalization(self, file: str):
        """
        Visualize results as needed for Section 5.3 of the paper
        :param file: experiments.json
        """
        # Load results
        with open(file) as json_file:
            content = json.load(json_file)
        rc('text', usetex=True)
        titles = ['Trained on $C_{TR10}$', 'Trained on $C_{TR20}$']
        for title, (min_speakers, max_speakers) in zip(titles, [(1, 10), (1, 20)]):
            plt.figure(dpi=200)
            ax = plt.gca()
            colors = ['#f0a804', '#FFDB37', '#0014cc', '#4D61FF']
            legends = ['$C_{te10}$ LOG\_STFT', '$C_{te10}$ MFCC', '$C_{te20}$ LOG\_STFT', '$C_{te20}$ MFCC', ]
            x = [str(i) for i in range(1, 21)]
            ys = []
            mean_maes = []
            for set in ['1_to_10', '1_to_20']:
                for feature_type in [TrainSetGenerator.FEATURE_TYPE_LOG_STFT, TrainSetGenerator.FEATURE_TYPE_MFCC]:
                    data = content[f'train_{min_speakers}_{max_speakers}'][feature_type][set]
                    ys.append([data[i] for i in x if i in data])
                    mean_maes.append(data['1_to_20'])
            for legend, color, y, mean_mae in zip(legends, colors, ys, mean_maes):
                plt.plot(x[:len(y)], y, label=legend, color=color)
                plt.scatter(9.5, mean_mae, color=color, zorder=10)
            plt.title(title)
            plt.ylabel('MAE')
            plt.xlabel('Label')
            plt.legend(loc='upper right')
            plt.ylim(0, 10)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.show()

    def visualize_libricount_generalization(self, file: str):
        """
        Visualize results as needed for Section 5.2 of the paper
        :param file: experiments.json
        """
        # Load results
        with open(file) as json_file:
            content = json.load(json_file)
        rc('text', usetex=True)
        titles = ['Trained on $C_{TR10}$', 'Trained on $C_{TR20}$']
        for title, (min_speakers, max_speakers) in zip(titles, [(1, 10), (1, 20)]):
            plt.figure(dpi=200)
            ax = plt.gca()
            colors = ['#f0a804', '#FFDB37', '#0014cc', '#4D61FF']
            legends = ['$C_{te10}$ LOG\_STFT', '$C_{te10}$ MFCC', '$L_{te10}$ LOG\_STFT', '$L_{te10}$ MFCC', ]
            x = [str(i) for i in range(1, 11)]
            ys = []
            mean_maes = []
            for set in ['1_to_10', 'libri']:
                for feature_type in [TrainSetGenerator.FEATURE_TYPE_LOG_STFT, TrainSetGenerator.FEATURE_TYPE_MFCC]:
                    data = content[f'train_{min_speakers}_{max_speakers}'][feature_type][set]
                    ys.append([data[i] for i in x])
                    mean_maes.append(data['1_to_10'])
            for legend, color, y, mean_mae in zip(legends, colors, ys, mean_maes):
                plt.plot(x, y, label=legend, color=color)
                plt.scatter(4.5, mean_mae, color=color, zorder=10)
            plt.title(title)
            plt.ylabel('MAE')
            plt.xlabel('Label')
            plt.legend(loc='upper right')
            plt.ylim(0, 10)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.show()

    def feature_comparison_csv(self, filename):
        """
        Generate a comparison csv, as needed for Section 5.1 of the paper
        :param filename:  The file comparison.json, created by Experimenter.run()
        """
        with open(filename) as file:
            content = json.load(file)
        file = open('feature_comparison.csv', 'w', newline='')
        writer = csv.writer(file, delimiter=';')
        values_to_compare = ['MAE C_{tr}', 'MAE C_{va}', 'Loss_{tr}', 'Loss_{va}', 'LR', 's / Epoch', '#Epochs']
        writer.writerow([''] + flatten([[v, v] for v in values_to_compare]))
        writer.writerow([''] + [10, 20] * len(values_to_compare))
        feature_types = TrainSetGenerator.FEATURE_OPTIONS
        for feature_type in feature_types:
            row = [feature_type]
            for measure in ['mean_absolute_error', 'val_mean_absolute_error', 'loss', 'val_loss', 'lr']:
                for train_type in ['train_1_10', 'train_1_20']:
                    row.append(min(content[train_type][feature_type]['history'][measure]))
            row.append(self.__mean_wo_outliers(content['train_1_10'][feature_type]['history']['timer']))
            row.append(self.__mean_wo_outliers(content['train_1_20'][feature_type]['history']['timer']))
            row.append(len(content['train_1_10'][feature_type]['history']['timer']))
            row.append(len(content['train_1_20'][feature_type]['history']['timer']))
            writer.writerow(row)
        file.close()

    def __mean_wo_outliers(self, data):
        """
        Get the mean of list of values, excluding outliers. Used to compute mean time per epoch. Outliers occur
        due to waiting times on the server, we do not want to include this in the computation of the mean.
        :param data:  The datapoint
        :return:  The mean
        """
        # data =
        z_scores = stats.zscore(data)
        valid_values = np.array(data)[[i for i, x in enumerate(z_scores) if z_scores[i] > -.5 and z_scores[i] < 0]]
        return np.mean(valid_values)

    def test_networks(self):
        """
        Test the networks saved by run(). Save to json
        """
        data = self.__get_test_data()
        result = {}
        for (min_speakers, max_speakers) in [[1, 10], [1, 20]]:
            result_for_trainset = {}
            for feature_type in self.feature_options:
                result_for_feature = {}
                # Load best performing model
                network = RNN()
                name = f'./trained_networks_with_augmentation/rnn_train_{min_speakers}_{max_speakers}/{feature_type}'
                network.load_from_file(name)

                # Test performance
                for test_name, test_data_current in data.items():
                    x, y = test_data_current['x'], test_data_current['y']
                    result_for_feature[test_name] = self.__test_net(network, x, y, feature_type)

                result_for_trainset[feature_type] = result_for_feature
            result[f'train_{min_speakers}_{max_speakers}'] = result_for_trainset
        with open('experiment_networks_tested.json', 'w+') as fp:
            json.dump(result, fp)
        return result

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
