""" Test files for functions that import the emoji data from the csv and prepare it
    for the neural network """

import math
import pandas as pd
import data_load_utils as util


def test_filter_text_for_handles_does_what_it_says():
    """ check filters both handles and chars """
    sample_series = pd.Series(
        ['some text with a @handle',
         'more @text with handle',
         'I`m an undesirable character'],
        name='text')

    compare_series = pd.Series(
        ['some text with a',
         'more with handle',
         'Im an undesirable character'],
        name='text')

    filtered_series = util.filter_text_for_handles(sample_series)
    assert filtered_series[0] == compare_series[0]  # handles
    assert filtered_series[1] == compare_series[1]  # handles
    assert filtered_series[2] == compare_series[2]  # characters
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

    test_text = util.pad_text(my_dict['text'], length=160)

    for i, srs in enumerate(series):
        assert srs + next_chars[i] == test_text[(i*stp):(i*stp)+window+1]


def test_one_hot_encoding_decodes_again():
    my_dict = {'text': "red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
               'emoji': ":rainbow:"}
    my_series = pd.Series(my_dict)
    window = 40
    stp = 3

    series, next_chars = util.get_series_data_from_tweet(my_series,
                                                         window_size=window,
                                                         step=stp)
    # chars, _ = util.get_unique_chars_list(series)
    chars, _ = util.get_universal_chars_list()
    x, y = util.get_x_y_bool_arrays(series, next_chars)

    for i in range(10):
        reconverted = util.x_y_bool_array_to_sentence(x, y, chars, i)
        assert reconverted == util.pad_text(my_dict['text'], length=160)[(i*stp):(i*stp)+window+1]


def test_one_hot_encoding_with_separate_functions_decodes():
    """ tests util.get_x_bool_arrays and get_y_bool_arrays get arrays that corespond """

    my_dict = {'text': "red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
               'emoji': ":rainbow:"}
    my_series = pd.Series(my_dict)
    window = 40
    stp = 3

    series, next_chars = util.get_series_data_from_tweet(my_series,
                                                         window_size=window,
                                                         step=stp)
    # chars, _ = util.get_unique_chars_list(series)
    chars, char_index = util.get_universal_chars_list()
    x = util.get_x_bool_array(series, chars, char_index)
    y = util.get_y_bool_array(next_chars, char_index)

    for i in range(1, len(y), 1):  # test for all values of y, why not
        reconverted = util.x_y_bool_array_to_sentence(x, y, chars, i)
        assert reconverted == util.pad_text(my_dict['text'], length=160)[(i*stp):(i*stp)+window+1]

    # test that the optional separator works
    reconverted = util.x_y_bool_array_to_sentence(x, y, chars, i, separator=True)
    original = util.pad_text(my_dict['text'], length=160)[(i*stp):(i*stp)+window+1]
    assert reconverted == original[0:-1] + ':' + original[-1]


def test_pad_text_returns_160():
    """ tests pad_text() returns exactly the expected number of characters """
    assert len(util.pad_text("a", length=160)) == 160
    assert len(util.pad_text("", length=160)) == 160
    assert len(util.pad_text("a" * 200, length=160)) == 160


def test_pad_text_returns_whitespace():
    assert util.pad_text("", length=160) == " " * 160


def test_convert_tweet_to_xy():
    """ test the wrapping function converts from pd.DataFrame """

    my_dict = {'text':
               ["red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
                "sweet dreams are made of this, who am I to disagree, travel the world and the even seas, every body's looking for someone"],
               'emoji':
               [":rainbow:",
                ":gay_pride_flag:"]}

    truncate_length = [90, 135, 160, 200]  # 160
    window_size = [10, 37, 40, 55]
    step = [2, 3, 5, 7]  # 3
    chars, _ = util.get_universal_chars_list()

    for t in truncate_length:
        for w in window_size:
            for s in step:

                my_data = pd.DataFrame(my_dict)
                x, y = util.convert_tweet_to_xy(my_data,
                                                length=t,
                                                window_size=w,
                                                step=s)

                # check dimensions are as expected
                assert x.shape[1] == w  # indexes over position in the window
                assert x.shape[2] == len(chars)  # indexes over one-hot encoding
                assert y.shape[0] == x.shape[0]
                assert y.shape[1] == len(chars)

                # check number of examples per tweet is given by math.ceil((length - window_size)/step)
                assert x.shape[0] == math.ceil((t - w) / s) * 2

                # check one-hot encoding decodes again
                for i in range(5):
                    util.x_y_bool_array_to_sentence(x[i], y[i], chars, position=i) == util.pad_text(
                        my_dict['text'][0], length=160)[(i*s):(i*s)+w+1]


def test_convert_tweet_to_xy_generator():
    """ test the generator function """

    my_dict = {'text':
               ["red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
                "sweet dreams are made of this, who am I to disagree, travel the world and the even seas, every body's looking for someone"],
               'emoji':
               [":rainbow:",
                ":gay_pride_flag:"]}

    truncate_length = [90, 135, 160, 200]  # 160
    window_size = [10, 37, 40, 55]
    step = [2, 3, 5, 7]  # 3
    chars, _ = util.get_universal_chars_list()

    for t in truncate_length:
        for w in window_size:
            for s in step:

                my_data = pd.DataFrame(my_dict)
                my_generator = util.convert_tweet_to_xy_generator(my_data,
                                                                  length=t,
                                                                  window_size=w,
                                                                  step=s,
                                                                  batch_size=1)

                for i in range(len(my_data)):
                    (x, y) = next(my_generator)

                    # check dimensions are as expected
                    assert x.shape[0] == math.ceil((t - w) / s) * 1
                    assert x.shape[1] == w  # indexes over position in the window
                    assert x.shape[2] == len(chars)  # indexes over one-hot encoding
                    assert y.shape[0] == math.ceil((t - w) / s) * 1
                    assert y.shape[1] == len(chars)

                    # check number of examples per tweet is given by math.ceil((length - window_size)/step)

                    # check one-hot encoding decodes again
                    for i in range(5):
                        util.x_y_bool_array_to_sentence(x[i], y[i], chars, position=i) == util.pad_text(
                            my_dict['text'][0], length=160)[(i*s):(i*s)+w+1]


def test_convert_tweet_to_xy_generator_emojis():
    """ test the generator function works for emojis"""

    my_dict = {'text':
               ["red and yellow and pink and green, orange and purple and blue, I can sing a rainbow, sing a rainbow, sing a rainbow too",
                "sweet dreams are made of this, who am I to disagree, travel the world and the even seas, every body's looking for someone"],
               'emoji':
               [":rainbow:",
                ":gay_pride_flag:"]}

    truncate_length = [90, 135, 160, 200]  # 160
    window_size = [10, 37, 40, 55]
    step = [2, 3, 5, 7]  # 3
    chars, _ = util.get_universal_chars_list()

    for t in truncate_length:
        for w in window_size:
            for s in step:

                my_data = pd.DataFrame(my_dict)

                emojis, emoji_index = util.get_emojis_list(my_data['emoji'])
                assert emojis
                my_generator = util.convert_tweet_to_xy_generator(my_data,
                                                                  length=t,
                                                                  window_size=w,
                                                                  step=s,
                                                                  batch_size=1,
                                                                  emoji_set=emojis)

                for i in range(len(my_data)):
                    (x, y) = next(my_generator)
                    [x_text, x_emoji] = x
#                    ([x_text, x_emoj], y) = next(my_generator)

                    # check dimensions are as expected
                    assert x_text.shape[0] == math.ceil((t - w) / s) * 1
                    assert x_text.shape[1] == w  # indexes over position in the window
                    assert x_text.shape[2] == len(chars)  # indexes over one-hot encoding
                    assert x_emoji.shape[0] == math.ceil((t - w) / s) * 1
                    assert x_emoji.shape[1] == len(emojis)
                    assert y.shape[0] == math.ceil((t - w) / s) * 1
                    assert y.shape[1] == len(chars)

                    # check number of examples per tweet is given by math.ceil((length - window_size)/step)

                    # check one-hot encoding decodes again
                    for i in range(5):
                        util.x_y_bool_array_to_sentence(x_text[i], y[i], chars, position=i) == util.pad_text(
                            my_dict['text'][0], length=160)[(i*s):(i*s)+w+1]
