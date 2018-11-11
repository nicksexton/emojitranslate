""" Functions that load downloaded emoji data and prepare train/dev/test sets for NNs """


import os
import string
import pandas as pd
import numpy as np
import emoji


CHARACTERS = """ '",.\\/|?:;@'~#[]{}-=_+!"£$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890"""


def read_tweet_data(path):
    """" loads the csv (path) containing text and emoji data
    returns a pandas dataframe containing line number, text, and emoji """
    data = pd.read_csv(path)
    data = data.loc[:, ['text', 'emoji']]  # should contain two labelled columns
    return data


def filter_tweets_min_count(tweets, min_count=1000):
    """ loads an m x 3 pandas dataframe (cols line number, text, emoji) and returns
    filtered list with only emojis with >min_count examples """
    return tweets.groupby('emoji').filter(lambda c: len(c) > min_count)


def filter_text_for_handles(text, chars=CHARACTERS):
    """ takes an pd.Series of text, removes twitter handles from
    text data - all text preceded by @ and then all characters not contained in 
    universal set"""

    def filter_handles(txt): return ' '.join(
        word for word in txt.split(' ') if not word.startswith('@'))

    def filter_chars(txt): return ''.join([c for c in txt if c in chars])

    def filter(txt): return filter_chars(filter_handles(txt))

    return text.apply(filter)


def pad_text(text, length=160):
    """ pads text with preceding whitespace and/or truncates tweet to 160 characters """

    if len(text) > length:
        return text[0:length]

    padded_text = ' ' * (length - len(text)) + text
    return padded_text


def filter_text(text):
    """ a wrapper for the previous filtering functions"""
    return filter_text_for_chars(filter_text_for_handles(text))


def get_series_data_from_tweet(tweet, length=160, window_size=40, step=3):
    """ input (tweet) is a pd.Series, a row of a pd.DataFrame
    returns corresponding lists sentences (of length window_size)
    and next_chars (single character). """

    sentences = []
    next_chars = []

    # pad all tweets to 160 characters
    # padded_text = ' ' * (160-tweet_length) + tweet['text']
    padded_text = pad_text(tweet['text'], length=length)

    for i in range(0, length - window_size, step):
        sentences.append(padded_text[i:i+window_size])
        next_chars.append(padded_text[i+window_size])

    return (sentences, next_chars)


def get_unique_chars_list(list_strings):
    """ takes list of strings, returns dict of all characters """

    one_big_string = ' '.join(list_strings)

    chars = sorted(list(set(one_big_string)))
    # print('Unique chars: ', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)

    return chars, char_indices


def get_universal_chars_list():
    """ gets a universal set of text characters and basic punctuation, suitable for using
    on all tweets. returns set of characters and the index. """

    return get_unique_chars_list(CHARACTERS)


def get_x_y_bool_arrays(sentences, next_chars):
    """ takes the list of strings (sentences) and list of next_chars, and
    one-hot encodes them using Boolean type, returns as arrays of x, y.
    Now replaced by get_x_bool_array and get_y_bool_array as vectorisable versions
    that work over a pd.Series"""
    print("Deprecated! Use get_x_bool_array or get_y_bool_array instead")
    chars, char_index = get_universal_chars_list()

    text_x = np.zeros((len(sentences), len(sentences[0]),
                       len(chars)), dtype=np.bool)
    text_y = np.zeros((len(sentences), len(char_index)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for pos, char in enumerate(sentence):
            text_x[i, pos, char_index[char]] = 1
        text_y[i, char_index[next_chars[i]]] = 1

    return (text_x, text_y)


def get_x_bool_array(sentence, chars, char_index):
    """ similar to get_x_y_bool_arrays() but operates on a single
    sentence and returns a one-hot encoded bool array (dims len(sentence) x len(chars)).
    Series chars is a list of recognised characters and char_index is the corresponding index"""

    # chars, char_index = get_unique_chars_list(sentence)

    text_x = np.zeros((len(sentence), len(sentence[0]),
                       len(chars)), dtype=np.bool)
    # text_y = np.zeros((len(sentences), len(char_index)), dtype=np.bool)
    for i, s in enumerate(sentence):
        for pos, char in enumerate(s):
            text_x[i, pos, char_index[char]] = 1
        # text_y[i, char_index[next_chars[i]]] = 1

    return np.asarray(text_x)


def get_y_bool_array(next_chars, char_index):
    """ similar to get_x_y_bool_arrays() but operates on a single
    sentence and returns a one-hot encoded bool array only (one dimension of size len(chars)).
    Series chars is a list of recognised characters and char_index is the corresponding index"""

    # Pass in a global list/index of characters so it's the same encoding for all tweets
    # chars, char_index = get_unique_chars_list(sentence)

    # text_x = np.zeros((len(sentence), len(sentence[0]),
    #                   len(chars)), dtype=np.bool)
    text_y = np.zeros((len(next_chars), len(char_index)), dtype=np.bool)
    for i in range(len(next_chars)):
        text_y[i, char_index[next_chars[i]]] = 1

    return np.asarray(text_y)


def x_y_bool_array_to_sentence(text_x, text_y, chars, position=0):
    """ converts one-hot encoded arrays text_x, text_y back to human
    readable, for debug purposes """

    def bool_array_to_char(bool_array, chars):
        return chars[np.argmax(bool_array.astype(int))]

    def decode_line(text_x, chars):
        string = []
        for i in range(text_x.shape[0]):
            string.append(bool_array_to_char(text_x[i], chars))
        return string

    def decode_example(text_x, text_y):
        # decodes x, y from array type back into english
        return(''.join(decode_line(text_x, chars)) +  # decode x
               bool_array_to_char(text_y, chars))   # decode y

    return decode_example(text_x[position], text_y[position])


def convert_tweet_to_xy(tweet, length=160, window_size=40, step=3):
    """ converts a tweet (pd DataFrame with 'text' field) to x, y text pairs, where x is
    window_size character moving window over the text, and y is the expected next character.
    outputs an ndarray of dims (m, window_size, characters) where m is the final number of 
    training examples and characters is the number of characters in the set (78 by default) """

    # apply the function to split each tweet into multiple windows of 40 chars and
    # a corresponding n_char
    # spits out a list of (x, y) tuples which is a real headache but we can fix it

    assert length > window_size

    zipped = tweet.apply(
        lambda x: get_series_data_from_tweet(x, length=length, window_size=window_size, step=step),
        axis=1)

    (x_tuple, y_tuple) = zip(*zipped)  # unzips the tuples into separate tuples of x, y

    # get the universal character set and the corresponding index
    chars_univ, char_idx_univ = get_universal_chars_list()

    x_bool = pd.Series(x_tuple).apply(lambda x: get_x_bool_array(x, chars_univ, char_idx_univ))

    y_bool = pd.Series(y_tuple).apply(lambda x: get_y_bool_array(x, char_idx_univ))

    x_dims = (len(x_bool),           # indexes over tweets
              # indexes over different sentence windows ((160 - window_size) / step = 40)
              x_bool[0].shape[0],
              x_bool[0].shape[1],    # indexes over characters in the window (window_size = 40)
              x_bool[0].shape[2])    # one-hot encoding for each character (78)

    y_dims = (len(y_bool),           # indexes over tweets
              y_bool[0].shape[0],    # indexes over different sentence windows (40)
              y_bool[0].shape[1])    # one-hot encoding for each character (78)

    # allocate space for the array
    x_arr = np.zeros(shape=x_dims)
    y_arr = np.zeros(shape=y_dims)

    for i, twit in enumerate(x_bool):
        x_arr[i] = twit

    for i, nchar in enumerate(y_bool):
        y_arr[i] = nchar

    # finally, reshape into a (m, w, c) array
    # where m is training example, w is window size,
    # c is one-hot encoded character
    x_fin = x_arr.reshape(x_arr.shape[0] * x_arr.shape[1], x_arr.shape[2], x_arr.shape[3])

    # y is a (m, c) array, where m is training example and c is one-hot encoded character
    y_fin = y_arr.reshape(y_arr.shape[0] * y_arr.shape[1], y_arr.shape[2])

    return x_fin, y_fin
