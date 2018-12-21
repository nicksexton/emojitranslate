""" Test file for functions that import/preprocess twitter data for the seq2seq model"""


import pandas as pd
import numpy as np
import data_load_seq2seq_utils as s2s_util
import data_load_utils as util


def test_filter_text_filters_for_newlines():
    """ check filters both handles and chars """
    sample_series = pd.Series(
        ['some text with a \nnewline',
         'more \ntext with newline',
         'I`m an undesirable character'],
        name='text')

    compare_series = pd.Series(
        ['some text with a newline',
         'more text with newline',
         'Im an undesirable character'],
        name='text')

    filtered_series = util.filter_text_for_handles(sample_series)
    assert filtered_series[0] == compare_series[0]  # newline
    assert filtered_series[1] == compare_series[1]  # newline
    assert filtered_series[2] == compare_series[2]  # characters
    return


def test_xy_generator():

    my_dict = {'text':
               ["red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
                "sweet dreams are made of this, who am I to disagree, travel the world and the even seas, every body's looking for someone"],
               'emoji':
               [":rainbow:",
                ":gay_pride_flag:"]}
    my_data = pd.DataFrame(my_dict)

    # copy the data 32 times so we have 64 rows
    my_data_repeat = pd.concat([my_data]*32, ignore_index=True)

    gen = s2s_util.xy_generator(my_data_repeat)

    (x, y) = gen.__next__()

    assert (x.shape[0] == 64)  # check generator returns an array with the right number of rows

    # check x and y are identical but shifted once on axis 1
    for i in range(x.shape[0]):
        for j in range(1, x.shape[1]):
            for k in range(x.shape[2]):
                assert x[i, j, k] == y[i, j-1, k]


def test_emoji_gen():
    """ tests that xy_generator also makes an emoji generator """

    my_dict = {'text':
               ["red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
                "sweet dreams are made of this, who am I to disagree, travel the world and the even seas, every body's looking for someone"],
               'emoji':
               [":rainbow:",
                ":gay_pride_flag:"]}
    my_data = pd.DataFrame(my_dict)

    # copy the data 32 times so we have 64 rows
    my_data_repeat = pd.concat([my_data]*32, ignore_index=True)

    gen = s2s_util.xy_generator(my_data_repeat)

    emojis = sorted(list(set(my_data_repeat['emoji'])))
    emoji_idx = dict((emoji, emojis.index(emoji)) for emoji in emojis)

    gen = s2s_util.xy_generator(my_data_repeat, emoji_indices=emoji_idx)

    (emoj, x, y) = gen.__next__()
    # check emoji array has the right batch size:
    assert emoj.shape[0] == 64

    # check emoji array is the right shape
    assert emoj.shape[1] == len(emoji_idx)

    # check each row of emoji array sums to exactly one
    for m in range(emoj.shape[0]):
        assert np.sum(emoj[m, :]) == 1
