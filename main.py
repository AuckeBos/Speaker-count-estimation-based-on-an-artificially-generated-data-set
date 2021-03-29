from DataLoader import DataLoader
from RNN import RNN

train_src_dir = './data/TIMIT/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMIT/wavfiles16kHz/TEST'

train_dest_dir = './data/CUSTOM_TIMIT/TRAIN'
test_dest_dir = './data/CUSTOM_TIMIT/TEST'

if __name__ == '__main__':
    # Load data from filesystem
    data_loader = DataLoader(train_src_dir, test_src_dr, train_dest_dir, test_dest_dir)
    data_loader.save_to_file = False
    data_loader.load_from_file = True
    train_x, train_y, test_x, test_y = data_loader.load_data()
    file = './exponentialactivation_wopooling_nodense40_nologscale'
    # Train network
    net = RNN()
    net.save_to_file = file
    # net.load_from_file(file)
    net.train(train_x, train_y)
    net.test(test_x, test_y)
