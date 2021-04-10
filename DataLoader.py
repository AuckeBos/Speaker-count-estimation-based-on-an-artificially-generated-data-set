import glob
import os
import random
from pathlib import Path

import numpy as np
import sklearn
from scipy.io import wavfile

from helpers import write_log


class DataLoader:
    """
    Class responsible for loading filenames from filesystem
    - Can generate the CUSTOM_TIMIT dataset by merging wav files from TIMIT
    - Can load datasets from files. Datasets are simply lists of filenames and a corresponding list of speaker counts
    """
    # Location of data
    train_src_dir: str
    test_src_dir: str

    # Location of generated data
    train_dest_dir: str
    test_dest_dir: str

    # Create wav files of min_speakers - max_speakers concurrent speakers
    min_speakers = 1
    max_speakers = 20

    # Files are sampled at 16kHz
    sampled_at = 16000

    # If true, always regenerate data, even if dirs already exist
    force_recreate = False

    # To reproduce
    random_state = 1337

    def __init__(self, train_src_dir: str, test_src_dr: str, train_dest_dir: str, test_dest_dir: str):
        """
        Save the src and dest dir
        :param train_src_dir: Dir that contains the original Timit files
        :param test_src_dr: Dir that contains the original Timit files
        :param train_dest_dir: Dir that will contain the generated merged wav files
        :param test_dest_dir:  Dir that will contain the generated merged wav files
        """
        self.train_src_dir = train_src_dir
        self.test_src_dir = test_src_dr
        self.train_dest_dir = train_dest_dir
        self.test_dest_dir = test_dest_dir

    def load_data(self):
        """
        Load data from train_dest_dir and test_dest_dir.
        - If self.load_from_file, load dataset from .npy files instead
        - If dest dirs do not exist, generated meged wav files
        :param force_recreate:
        :return train_x, train_y, test_x, test_y
        """
        if self.force_recreate or not os.path.exists(self.train_dest_dir) or not os.path.exists(self.test_dest_dir):
            self.__generate_datasets()
        return self.__load_datasets()

    @staticmethod
    def load_libricount(dir: str):
        """
        Load the LibriCount dataset
        :param dir: The dir where the set is stored
        :return: (X (filenames), Y (speaker counts))
        """
        files = glob.glob(dir + '/*.wav')
        X, Y = [], []
        for file in files:
            filename = os.path.basename(file)
            y = int(filename[0])
            # Skip recordings w/o speakers
            if y == 0:
                continue
            Y.append(y)
            X.append(file)
        return (np.array(X), np.array(Y))

    def __load_datasets(self):
        """
        Load datasets:
        - Loop over all train and test files
        - Create lists of filenames and speaker counts
        :return (train_x (filenames), train_y (speaker_counts)), (test_x, test_y)
        """
        train_x, train_y, test_x, test_y = [], [], [], []
        for y in range(self.min_speakers, self.max_speakers + 1):
            train_dir = f'{self.train_dest_dir}/{y}'
            test_dir = f'{self.test_dest_dir}/{y}'
            train_files = glob.glob(train_dir + '/*.wav')
            train_x.extend(train_files)
            train_y.extend([y] * len(train_files))

            test_files = glob.glob(test_dir + '/*.wav')
            test_x.extend(test_files)
            test_y.extend([y] * len(test_files))
        return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))

    def __generate_datasets(self):
        """
        Generate train and test datasets, by merging wav files from source dir
        """
        write_log('Generating data')
        dirs = [(self.train_src_dir, self.train_dest_dir, True), (self.test_src_dir, self.test_dest_dir, False)]
        for (src_dir, dest_dir, shuffle) in dirs:
            files = glob.glob(src_dir + '/*.WAV')
            num_records_per_count = len(files) // self.max_speakers
            data = [record for (_, record) in [wavfile.read(wav) for wav in files]]
            for i in range(self.min_speakers, self.max_speakers + 1):
                write_log(f'Generating files for {i} concurrent speakers')
                self.__create_concurrent_speakers(data, dest_dir, i, num_records_per_count, shuffle)
        write_log('Data generated')

    def __create_concurrent_speakers(self, train, dest_dir, num_speakers, num_records, shuffle):
        """
        Generate wav files with concurrent speakers
        :param shuffle: If true, shuffle dataset before partitioning
        :param dest_dir:  The dir to save the new files in
        :param train:  The original wav files
        :param num_speakers:  The number of speakers per sample
        :param num_records:  The number of samples to create
        """
        # Save files in subdir with name 'num_speakers'
        dest_dir += f'/{num_speakers}'
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if shuffle:
            random.Random(self.random_state).shuffle(train)
        # Generate partitions of length num_speakers
        partitions = [train[i:i + num_speakers] for i in range(0, num_records * num_speakers, num_speakers)]

        for i, partition in enumerate(partitions):
            # Pad to size of longest file
            pad_to = len(max(partition, key=len))
            partition = np.array([np.pad(x, (0, pad_to - len(x))) for x in partition])
            dest_filename = f'{dest_dir}/{i}.wav'
            with open(dest_filename, 'wb+') as dest_file:
                wavfile.write(dest_file, self.sampled_at, partition.T)

            # Normlize the resulting wav file, to avoid clipping
            # wav = AudioSegment.from_file(dest_filename)
            # wav = effects.normalize(wav)
            # wav.export(dest_filename)
