""" Functions that load downloaded tweet/emoji data into a data frame and process it 
    into numpy tran/dev/test sets for a Seq2Seq model.

    x is (tweet_length, character_set_size) sized ndarray"""

import os
import string
import pandas as pd
import numpy as np
import data_load_utils as prev_util


def xy_generator(tweets, batch_size=64, sequence_length=161):
    """ Generator function that returns an (X, Y) tuple, where X and Y are a numpy 
    array of shape (batch_size, sequence_length, len(char_indices) 
    sequence_length is 160 (longest tweet) + newline """

    # NB remeber to append \n to all sequences and include it in char_indices

    # Iterate over the dataset
    batch_num = 0
    n_batches = int(tweets.shape[0] / batch_size)  # terminate after last full batch for now

    chars_univ, char_idx_univ = prev_util.get_universal_chars_list()

    x_dims = (batch_size, sequence_length, len(char_idx_univ))
    x_arr = np.zeros(shape=x_dims)

    while batch_num < n_batches:  # in case tweets < batch_size

        # slice the batch
        this_batch = tweets.iloc[(batch_num*batch_size):(batch_num+1)*batch_size]

        for m in range(batch_size):
            for i, char in enumerate(this_batch.iloc[m].loc['text']):
                x_arr[m, i, char_idx_univ[char]] = 1

        yield x_arr


def emoji_generator(emoji_set=None):
    return 'temp'
