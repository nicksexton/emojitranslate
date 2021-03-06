""" Functions that load downloaded tweet/emoji data into a data frame and process it
    into numpy tran/dev/test sets for a Seq2Seq model.

    x is (tweet_length, character_set_size) sized ndarray"""

import os
import string
import pandas as pd
import numpy as np
import data_load_utils as prev_util


# including newline character in CHARACTERS
CHARACTERS = """\n '",.\\/|?:;@'~#[]{}-=_+!"£$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890"""
CHARACTERS_NO_NEWLINE = """ '",.\\/|?:;@'~#[]{}-=_+!"£$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890"""


def get_unique_chars_list(list_strings):
    """ takes list of strings, returns dict of all characters """

    one_big_string = ' '.join(list_strings)

    chars = sorted(list(set(one_big_string)))
    char_indices = dict((char, chars.index(char)) for char in chars)

    return chars, char_indices


def get_universal_chars_list():
    """ gets a universal set of text characters and basic punctuation, suitable for using
    on all tweets. returns set of characters and the index. """

    return get_unique_chars_list(CHARACTERS)


def filter_text(text, chars=CHARACTERS_NO_NEWLINE):
    """ takes an pd.Series of text. 
    Filters it, removing twitter handles from text data - all text preceded by @ and then 
    all characters not contained in universal set. Also removes all newline characters"""

    def filter_handles(txt): return ' '.join(
        word for word in txt.split(' ') if not word.startswith('@'))

    def filter_chars(txt): return ''.join([c for c in txt if c in chars])

    def filter(txt): return filter_chars(filter_handles(txt))

    return text.apply(filter)


def xy_generator(tweets, batch_size=64, sequence_length=161, emoji_indices=None):
    """ Generator function that returns an (X, Y) tuple, where X and Y are a numpy
    array of shape (batch_size, sequence_length, len(char_indices)
    sequence_length is 160 (longest tweet) + newline """

    # NB remeber to append \n to all sequences and include it in char_indices

    # Iterate over the dataset
    batch_num = 0
    n_batches = int(tweets.shape[0] / batch_size)  # terminate after last full batch for now

    chars_univ, char_idx_univ = get_universal_chars_list()

    x_dims = (batch_size, sequence_length, len(char_idx_univ))
    x_arr = np.zeros(shape=x_dims)
    y_arr = np.zeros(shape=(batch_size, sequence_length, len(char_idx_univ))
                     )  # should it be sequence_length -1?
    if emoji_indices:
        # shape is (batch_size, sequence_length, tokens)
        emoji_arr = np.zeros(shape=(batch_size, 1, len(emoji_indices)))

    while batch_num < n_batches:  # in case tweets < batch_size

        # slice the batch
        this_batch = tweets.iloc[(batch_num*batch_size):(batch_num+1)*batch_size]

        for m in range(batch_size):
            for i, char in enumerate(this_batch.iloc[m].loc['text'] + '\n'):
                x_arr[m, i, char_idx_univ[char]] = 1
                if i > 0:
                    # y_arr is ahead by one character and omits starting character
                    y_arr[m, i-1, char_idx_univ[char]] = 1

            if emoji_indices:
                emoji_arr[m, 0, emoji_indices[this_batch.iloc[m].loc['emoji']]] = 1

        if emoji_indices:
            yield ([emoji_arr, x_arr], y_arr)
        else:
            yield (x_arr, y_arr)
