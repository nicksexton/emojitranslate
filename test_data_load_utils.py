""" Test files for functions that import the emoji data from the csv and prepare it
    for the neural network """


import pandas as pd
import data_load_utils as util


def test_filter_text_for_handles_does_what_it_says():
    sample_series = pd.Series(
        ['some text with a @handle',
         'more @text with handle'],
        name='text')

    compare_series = pd.Series(
        ['some text with a',
         'more with handle'],
        name='text')

    filtered_series = util.filter_text_for_handles(sample_series)
    assert filtered_series[0] == compare_series[0]
    assert filtered_series[1] == compare_series[1]
    return


def test_get_series_data_from_tweet_really_matches_nextchars():
    my_dict = {'text': "red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
               'emoji': ":rainbow:"}
    my_series = pd.Series(my_dict)
    window = 40
    stp = 3
    series, next_chars = util.get_series_data_from_tweet(my_series,
                                                         window_size=window,
                                                         step=stp)

    for i, srs in enumerate(series):
        assert srs + next_chars[i] == my_dict['text'][(i*stp):(i*stp)+window+1]


def test_one_hot_encoding_decodes_again():
    my_dict = {'text': "red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
               'emoji': ":rainbow:"}
    my_series = pd.Series(my_dict)
    window = 40
    stp = 3

    series, next_chars = util.get_series_data_from_tweet(my_series,
                                                         window_size=window,
                                                         step=stp)
    chars, _ = util.get_unique_chars_list(series)
    x, y = util.get_x_y_bool_arrays(series, next_chars)

    for i in range(10):
        reconverted = util.x_y_bool_array_to_sentence(x, y, chars, i)
        assert reconverted == my_dict['text'][(i*stp):(i*stp)+window+1]
