from DataLoader import DataLoader
from RNN import RNN

train_src_dir = './data/TIMITS/TIMIT/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMITS/TIMIT/wavfiles16kHz/TEST'

train_dest_dir = './data/TIMITS/2_5_speakers/TRAIN'
test_dest_dir = './data/TIMITS/2_5_speakers/TEST'

if __name__ == '__main__':
    # Load data from filesystem
    data_loader = DataLoader(train_src_dir, test_src_dr, train_dest_dir, test_dest_dir)
    data_loader.force_recreate = False
    data_loader.save_to_file = False
    data_loader.load_from_file = False
    train_x, train_y, test_x, test_y = data_loader.load_data()
    # libri_x, libri_y = data_loader.load_libricount('./data/LibriCount/test')
    # Train network
    file = 'rnn_max_5_speakers_with_maxpool'
    net = RNN()
    # net.use_mfcc()
    # net.save_to_file(file)
    # net.save_to_file = file
    # net.load_from_file(file)
    net.train(train_x, train_y)
    # net.compile_net()
    net.test(test_x, test_y)
