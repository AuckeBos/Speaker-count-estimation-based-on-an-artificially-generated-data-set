import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DataLoader import DataLoader
from Experimenter import Experimenter
from RNN import RNN

train_src_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMITS/TIMIT/wavfiles16kHz/TEST'

train_dest_dir = './data/experiments/1_to_10/train'
test_dest_dir = './data/experiments/1_to_10/test'

if __name__ == '__main__':
    # # Load data from filesystem
    # data_loader = DataLoader(train_src_dir, test_src_dr, train_dest_dir, test_dest_dir)
    # # data_loader.force_recreate = False
    # # data_loader.save_to_file = False
    # # data_loader.load_from_file = False
    # data_loader.max_speakers = 10
    # train_x, train_y, test_x, test_y = data_loader.load_data()
    # # libri_x, libri_y = data_loader.load_libricount('./data/LibriCount/test')
    # # Train network
    # # file = 'rnn_test'
    # net = RNN()
    # net.set_feature_type(RNN.FEATURE_TYPE_STFT)
    # # net.use_mfcc()
    # # net.save_to_file(file)
    # # net.save_to_file = file
    # net.load_from_file('./trained_networks/rnn_train_max_10/STFT')
    # # net.train(train_x, train_y)
    # # net.compile_net()
    # net.test(test_x, test_y)

    experimenter = Experimenter()
    experimenter.run()