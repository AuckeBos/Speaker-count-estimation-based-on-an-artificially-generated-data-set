import os

import pandas as pd
import scipy
from scipy.io import wavfile
import numpy as np
import random
import glob
from pathlib import Path
from pydub import AudioSegment
import sklearn
from pydub import AudioSegment, effects
import os
from helpers import write_log


class DataLoader:
    """
    Class responsible for loading the data
    - Can generate the Custom_Timit dataset by merging wav files from Timit
    - Can load the dataset from the generated file system dir
    - Can load the dataset from filesystems npy file (for testing purposes)

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

    # Pad to and cut of at five seconds, during dataset generation
    pad_to = sampled_at * 5

    # If true, the dataset will be saved to .npy files after its loaded
    save_to_file = False

    # If true, the dataset will not be loaded from wav files, but from .npy files
    load_from_file = False

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
        if self.force_recreate:
            self.__generate_datasets()
        if self.load_from_file:
            return self.__load_from_file()
        if not os.path.exists(self.train_dest_dir) or not os.path.exists(self.test_dest_dir):
            self.__generate_datasets()
        return self.__load_datasets()

    @staticmethod
    def load_libricount(dir: str):
        write_log('Loading LibriCount')
        files = glob.glob(dir + '/*.wav')
        X, Y = [], []
        for file in files:
            filename = os.path.basename(file)
            y = int(filename[0])
            # Skip recordings w/0 speakers
            if y == 0:
                continue
            Y.append(y)
            _, x = wavfile.read(file)
            X.append(x)

        # Pad to and cut off
        X = DataLoader.__pad(X)
        Y = np.array(Y)

        # Shuffle
        X, Y = sklearn.utils.shuffle(X, Y, random_state=DataLoader.random_state)
        write_log('Data loaded')
        return X, Y

    @staticmethod
    def __pad(data):
        """
        Pad and cut off to self.pad_to
        :param data:
        :return:
        """
        return np.array([np.pad(x, (0, max(DataLoader.pad_to - len(x), 0)))[:DataLoader.pad_to] for x in data])

    def __load_datasets(self):
        """
        Load datasets into memory
        - Save them to .npy files if self.save_to_file
        :return train_x, train_y, test_x, test_y
        """
        write_log('Loading data from WAVs')
        train_x, train_y, test_x, test_y = [], [], [], []
        for y in range(self.min_speakers, self.max_speakers + 1):
            train_dir = f'{self.train_dest_dir}/{y}'
            test_dir = f'{self.test_dest_dir}/{y}'
            train_files = glob.glob(train_dir + '/*.wav')
            test_files = glob.glob(test_dir + '/*.wav')

            current_train_x = np.array([record for (_, record) in [wavfile.read(wav) for wav in train_files]], dtype='object')
            current_test_x = np.array([record for (_, record) in [wavfile.read(wav) for wav in test_files]], dtype='object')
            # True for speaker_count > 1: Sum wave sounds
            if current_train_x[0].ndim > 1:
                current_train_x = np.array([np.sum(x, axis=1) for x in current_train_x], dtype='object')
                current_test_x = np.array([np.sum(x, axis=1) for x in current_test_x], dtype='object')

            current_train_y = [y] * len(current_train_x)
            current_test_y = [y] * len(current_test_x)

            train_x.extend(current_train_x)
            train_y.extend(current_train_y)
            test_x.extend(current_test_x)
            test_y.extend(current_test_y)
        # Pad to and cut off
        train_x = self.__pad(train_x)
        test_x = self.__pad(test_x)
        train_y, test_y = np.array(train_y), np.array(test_y)

        # Shuffle
        train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
        test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
        write_log('Data loaded')
        # Save to file if desired
        if self.save_to_file:
            self.__save_to_file(train_x, train_y, test_x, test_y)
        return train_x, train_y, test_x, test_y

    def __load_from_file(self):
        """
        Instead of generating the data, load it from filesystem
        :return:  The data
        """
        write_log('Loading data from npy files')
        sets = ['train_x', 'train_y', 'test_x', 'test_y']
        # For each set, try to load it and set as attr on self
        for set in sets:
            filename = f'{set}.npy'
            # If any file is not found, raise an error
            if not os.path.exists(filename):
                raise FileNotFoundError(filename)
            setattr(self, set, np.load(filename))
        # Return sets as tuple
        write_log('Data loaded')
        return map(tuple, [getattr(self, set) for set in sets])

    @staticmethod
    def __save_to_file(train_x, train_y, test_x, test_y):
        """
        Save loaded data to files
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        """
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
        np.save('test_x.npy', test_x)
        np.save('test_y.npy', test_y)
        write_log('Data saved to npy files')

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
