import os

from DataLoader import DataLoader
from RNN import RNN
from TrainSetGenerator import TrainSetGenerator
from Experimenter import Experimenter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_dir = './data/TIMITS/TIMIT_SUB/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMITS/TIMIT_SUB/wavfiles16kHz/TEST'
test_dest_dir = './data/testing_dataloader/1_to_10/test'

libri_dir = './data/LibriCount/test'

if __name__ == '__main__':
    # min_speakers = 1
    # max_speakers = 10
    # feature_type = TrainSetGenerator.FEATURE_TYPE_STFT
    #
    # # Load data from filesystem
    # data_loader = DataLoader(train_dir, test_src_dr, test_dest_dir)
    # data_loader.force_recreate = False
    # data_loader.min_speakers = min_speakers
    # data_loader.max_speakers = max_speakers
    # # Train network
    # # file = 'net_variable_generator'
    # train, (test_x, test_y) = data_loader.load_data()
    # libri_x, libri_y = data_loader.load_libricount(libri_dir)
    #
    # # Train and test network
    # file = 'testing_rnn_2'
    # net = RNN()
    # net.save_to_file(file)
    # net.train(train, min_speakers, max_speakers, feature_type)
    #
    # net.load_from_file(file)
    #
    # timit_results = net.test(test_x, test_y, feature_type)
    # libri_results = net.test(libri_x, libri_y, feature_type)

    experimenter = Experimenter()
    # experimenter.feature_comparison_csv('experiments_with_augmentation.json')
    # experimenter.visualize('experiments_with_augmentation.json')
    experimenter.visualize('experiments_with_augmentation.json')
    # experimenter.run()
    # test = DataLoader.load_libricount('./data/LibriCount/test')
    # test = ''

