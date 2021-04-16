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
    - Can generate the CUSTOM_TIMIT dataset by merging wav files from TIMIT. This is used to generate TEST sets.
    - Can load datasets from files. Datasets are simply lists of filenames and a corresponding list of speaker counts
    """
    # Location of data
    train_dir: str
    test_src_dir: str

    # Location of generated TEST data
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

    def __init__(self, train_dir: str, test_src_dr: str, test_dest_dir: str):
        """
        Save the src and dest dir
        :param train_dir: Dir that contains the original Timit train files
        :param test_src_dr: Dir that contains the original Timit test files
        :param test_dest_dir:  Dir that will contain the generated merged test wav files
        """
        self.train_dir = train_dir
        self.test_src_dir = test_src_dr
        self.test_dest_dir = test_dest_dir

    def load_data(self):
        """
        Load data from train_dest_dir and test_dest_dir.
        - If test dest dir does not exist, generated test set
        :return train, test_x, test_y
        """
        if self.force_recreate or not os.path.exists(self.test_dest_dir):
            self.__generate_test_set()
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
        :return train (filenames), (test_x, test_y)
        """
        train = glob.glob(self.train_dir + '/*.WAV')
        test_x, test_y = [], []
        for y in range(self.min_speakers, self.max_speakers + 1):
            dir = f'{self.test_dest_dir}/{y}'

            files = glob.glob(dir + '/*.wav')
            test_x.extend(files)
            test_y.extend([y] * len(files))
        return np.array(train), (np.array(test_x), np.array(test_y))

    def __generate_test_set(self):
        """
        Generate a test dataset, by merging wav files from source dir
        """

        # If yet generated, skip
        if os.path.exists(self.test_dest_dir):
            return

        write_log('Generating Test set')
        files = glob.glob(self.test_src_dir + '/*.WAV')
        num_records_per_count = len(files) // self.max_speakers
        data = [record for (_, record) in [wavfile.read(wav) for wav in files]]
        for i in range(self.min_speakers, self.max_speakers + 1):
            write_log(f'Generating test files for {i} concurrent speakers')
            self.__create_concurrent_speakers(data, self.test_dest_dir, i, num_records_per_count)
        write_log('Data generated')

    def __create_concurrent_speakers(self, data, dest_dir, num_speakers, num_records):
        """
        Generate wav files with concurrent speakers
        :param dest_dir:  The dir to save the new files in
        :param data:  The original wav files
        :param num_speakers:  The number of speakers per sample
        :param num_records:  The number of samples to create
        """
        # Save files in subdir with name 'num_speakers'
        dest_dir += f'/{num_speakers}'
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        # Shuffle the set
        random.Random(self.random_state).shuffle(data)
        # Generate partitions of length num_speakers
        partitions = [data[i:i + num_speakers] for i in range(0, num_records * num_speakers, num_speakers)]

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
