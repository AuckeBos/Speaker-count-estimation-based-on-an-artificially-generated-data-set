import os
import glob
from DataGenerator import DataGenerator
from VariableDataGenerator import VariableDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DataLoader import DataLoader
from Experimenter import Experimenter
from RNN import RNN

train_src_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMITS/TIMIT/wavfiles16kHz/TEST'

train_dest_dir = './data/experiments/1_to_10/train'
test_dest_dir = './data/experiments/1_to_10/test'

if __name__ == '__main__':
    # Load data from filesystem
    # data_loader = DataLoader(train_src_dir, test_src_dr, train_dest_dir, test_dest_dir)
    # data_loader.force_recreate = False
    # data_loader.max_speakers = 10
    # libri_x, libri_y = data_loader.load_libricount('./data/LibriCount/test')
    # Train network
    file = 'net_variable_generator'

    files = glob.glob(f'{train_src_dir}/*.WAV')
    net = RNN()
    net.save_to_file(file)
    net.train_variable_batches(files, VariableDataGenerator.FEATURE_TYPE_STFT)
    # net.load_from_file('./trained_networks/rnn_train_max_10/STFT')

    # (train_x, train_y), (test_x, test_y) = data_loader.load_data()
    # net.train(train_x, train_y, DataGenerator.FEATURE_TYPE_STFT)
    # feature_type = DataGenerator.FEATURE_TYPE_STFT
    # input_shape = DataGenerator.get_shape_for_type(feature_type)
    # net.compile_net(input_shape)
    # net.test(test_x, test_y, feature_type)

    # experimenter = Experimenter()
    # experimenter.visualize('experiments.json')
    # experimenter.test_networks()
    # experimenter.run()
