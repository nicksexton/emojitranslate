""" Functions that load downloaded emoji data and prepare train/dev/test sets for NNs """


import os
import pandas as pd
import numpy as np
import emoji


def read_tweet_data(path):
    """" loads the csv (path) containing text and emoji data
    returns a pandas dataframe containing line number, text, and emoji """
    data = pd.read_csv(path)
    data = pd.loc[:, ['text', 'emoji']]  # should contain two labelled columns
    return data


def filter_tweets_min_count(tweets, min_count=1000):
    """ loads an m x 3 pandas dataframe (cols line number, text, emoji) and returns
    filtered list with only emojis with >min_count examples """
    return tweets.groupby('emoji').filter(lambda c: len(c) > min_count)


def filter_text_for_handles(text):
    """ takes an pd.Series of text, removes twitter handles from
    text data - all text preceded by @ """

    def filter_handles(txt): return ' '.join(
        word for word in txt.split(' ') if not word.startswith('@'))

    return text.apply(filter_handles)


def get_series_data_from_tweet(tweet, window_size=40, step=3):
    """ input (tweet) is a pd.Series, a row of a pd.DataFrame
    returns corresponding lists sentences (of length window_size)
    and next_chars (single character). """

    sentences = []
    next_chars = []
    tweet_length = len(tweet['text'])

    for i in range(0, tweet_length - window_size, step):
        sentences.append(tweet['text'][i:i+window_size])
        next_chars.append(tweet['text'][i+window_size])

    return (sentences, next_chars)


def get_unique_chars_list(list_strings):
    """ takes list of strings, returns dict of all characters 
    ***** REMEMBER ***** to modify this code for multiple tweets """

    one_big_string = ' '.join(list_strings)

    chars = sorted(list(set(one_big_string)))
    print('Unique chars: ', len(chars))
    char_indices = dict((char, chars.index(char)) for char in chars)

    return chars, char_indices


def get_x_y_bool_arrays(sentences, next_chars):
    """ takes the list of strings (sentences) and list of next_chars, and
    one-hot encodes them using Boolean type, returns as arrays of x, y """

    chars, char_index = get_unique_chars_list(sentences)

    text_x = np.zeros((len(sentences), len(sentences[0]),
                       len(chars)), dtype=np.bool)
    text_y = np.zeros((len(sentences), len(char_index)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for pos, char in enumerate(sentence):
            text_x[i, pos, char_index[char]] = 1
        text_y[i, char_index[next_chars[i]]] = 1

    return (text_x, text_y)


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
