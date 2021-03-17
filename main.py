from DataLoader import DataLoader

train_src_dir = '.data/TIMIT/wavfiles16kHz/TRAIN'
test_src_dr = './data/TIMIT/wavfiles16kHz/TEST'

train_dest_dir = './data/CUSTOM_TIMIT/TRAIN'
test_dest_dir = './data/CUSTOM_TIMIT/TEST'

data_loader = DataLoader(train_src_dir, test_src_dr, train_dest_dir, test_dest_dir)
train_x, train_y, test_x, test_y = data_loader.load_data()
