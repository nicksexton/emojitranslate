""" Test file for functions that import/preprocess twitter data for the seq2seq model"""


import pandas as pd
import data_load_seq2seq_utils as util


def test_read_tweet_data_filters_headers():
    """ filter out column headers (rows where text='text' and emoji='emoji') """
    tweets = util.read_tweet_data('data/emojis_homemade.csv')
    assert tweets.shape[0] > 0  # check tweets dataframe created

    # run the filter again, check there's no hits
    filt = tweets['emoji'] == 'emoji'
    assert filt.any(axis=0) == False


def test_import_and_filter_go():
    """ import and filter text as it's intended to be used"""

    tweets = util.read_tweet_data('data/emojis_homemade.csv')
