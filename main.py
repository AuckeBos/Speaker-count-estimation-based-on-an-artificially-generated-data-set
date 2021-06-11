import os

from DataLoader import DataLoader
from RNN import RNN
from TrainSetGenerator import TrainSetGenerator
from Experimenter import Experimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Location of data files
train_dir = './data/TIMITS/TIMIT_SUB/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMITS/TIMIT_SUB/wavfiles16kHz/TEST'
test_dest_dir = './data/testing_dataloader/1_to_10/test'

libri_dir = './data/LibriCount/test'

# Change values to define what functions to run
TRAIN_AND_TEST_NETWORK = True
RUN_EXPERIMENTER = True
FEATURE_TYPE = TrainSetGenerator.FEATURE_TYPE_STFT


def train_and_test_network():
    """
    Train a neural network and test it. Can also train on other feature types,
    or run the experimenter to run different configurations
    """
    min_speakers = 1
    max_speakers = 10

    # Load data from filesystem
    data_loader = DataLoader(train_dir, test_src_dr, test_dest_dir)
    data_loader.force_recreate = False
    data_loader.min_speakers = min_speakers
    data_loader.max_speakers = max_speakers

    # Train network
    train, (test_x, test_y) = data_loader.load_data()
    libri_x, libri_y = data_loader.load_libricount(libri_dir)

    # Train and test network
    file = 'testing_rnn'
    net = RNN()
    net.save_to_file(file)
    net.train(train, min_speakers, max_speakers, FEATURE_TYPE)

    net.load_from_file(file)

    timit_results = net.test(test_x, test_y, FEATURE_TYPE)
    libri_results = net.test(libri_x, libri_y, FEATURE_TYPE)


def run_experimenter():
    """
    Run the experimenter. The Experimenter can also run other experiments, see Experimenter.
    :return:
    """
    experimenter = Experimenter()
    experimenter.run()


if __name__ == '__main__':
    if TRAIN_AND_TEST_NETWORK:
        train_and_test_network()
    if RUN_EXPERIMENTER:
        run_experimenter()
