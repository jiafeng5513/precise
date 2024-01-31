import numpy as np
import os
import linecache
import glob
"""
wakeup dataset download from [A far-field text-dependent speaker verification database for AISHELL Speaker Verification Challenge 2019](https://www.openslr.org/85/)
the wakeup word is "你好米雅(ni hao mi ya)"
this script random sample wav files from OpenSLR dataset to train and test sets 'wake-word' folder, resample it to 16000 sample rate

not-wakeup dataset download from 
    [Large-scale (1000 hours) corpus of read English speech](https://www.openslr.org/12/)
        and
    [A Free Chinese Speech Corpus Released by CSLT@Tsinghua University](https://www.openslr.org/18/)
this script random sample wav files from OpenSLR dataset to train and test sets 'not-wake-word' folder, resample it to 16000 sample rate
"""

SLR85_DATASET_PATH = '/mnt/data/dataset/HI-MIA-SLR85/train/SPEECHDATA/'
SLR18_DATASET_PATH = '/mnt/data/dataset/CSLT-Chinese-spech-SLR18/data_thchs30/train/'
training_data_p_dist_dir = '/home/anna/WorkSpace/celadon/demo-src/precise/training/data/wake-word'
training_data_n_dist_dir = '/home/anna/WorkSpace/celadon/demo-src/precise/training/data/not-wake-word'
val_data_dist_p_dir = '/home/anna/WorkSpace/celadon/demo-src/precise/training/data/test/wake-word'
val_data_dist_n_dir = '/home/anna/WorkSpace/celadon/demo-src/precise/training/data/test/not-wake-word'
num_train_files = 1000
num_val_files = 100


def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def sample_to_wakeup_words():
    """
    HI-MIA-SLR85 ---> wakeup data
    """
    training_data_list_file = os.path.join(SLR85_DATASET_PATH, "train.scp")
    total_samples_num = iter_count(training_data_list_file)

    def sample_and_trans(sample_nums, dst_path):
        training_sample_index = np.random.randint(1, total_samples_num, (sample_nums,))
        for sample_index in training_sample_index:
            wave_filename = get_line_context(training_data_list_file, sample_index)
            src_wave_file_path = os.path.join(SLR85_DATASET_PATH, wave_filename)
            dst_wave_file_path = os.path.join(dst_path, "p_" + os.path.basename(src_wave_file_path))
            os.popen("ffmpeg -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(src_wave_file_path, dst_wave_file_path))
        pass

    sample_and_trans(num_train_files, training_data_p_dist_dir)
    sample_and_trans(num_val_files, val_data_dist_p_dir)


def sample_to_not_wakeup_words():
    wav_filenames = glob.glob(os.path.join(SLR18_DATASET_PATH, '*.wav'))
    total_samples_num = len(wav_filenames)

    def sample_and_trans(sample_nums, dst_path):
        training_sample_index = np.random.randint(1, total_samples_num, (sample_nums,))
        for sample_index in training_sample_index:
            src_wave_file_path = wav_filenames[sample_index]
            dst_wave_file_path = os.path.join(dst_path, "n_" + os.path.basename(src_wave_file_path))
            os.popen("ffmpeg -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(src_wave_file_path, dst_wave_file_path))
        pass

    sample_and_trans(num_train_files, training_data_n_dist_dir)
    sample_and_trans(num_val_files, val_data_dist_n_dir)
    pass


if __name__ == '__main__':
    # sample_to_wakeup_words()
    sample_to_not_wakeup_words()
