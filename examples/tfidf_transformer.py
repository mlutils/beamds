from examples.example_utils import add_beam_to_path
add_beam_to_path()

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
import string
import random
from pathlib import Path

import sys
import time

from src.beam.data import BeamData
from src.beam.transformer import Transformer

from datasets import list_datasets, load_dataset
from pprint import pprint


class TFIDFTransformer(Transformer):

    def __init__(self, *args, state=None, strip_accents='ascii', stop_words='english', max_df=0.1, min_df=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.tfidf = TfidfVectorizer(strip_accents=strip_accents, stop_words=stop_words,
                                          max_df=max_df, min_df=min_df)
        if state is not None:
            self.load_state_dict(state)
            self.tfidf = self.state

    def transform_callback(self, bd, key=None, is_chunk=True, **kwargs):
        x = self.tfidf.transform(bd['text'].values)
        bd['tfidf'] = x
        return bd

    def fit(self, bd, **kwargs):
        self.tfidf.fit(bd['text'].values)
        self.state = self.tfidf


if __name__ == '__main__':
    dataset = load_dataset('ag_news')
    transformer = TFIDFTransformer(n_chunks=4, n_workers=2, split_by='index')

    data_train = BeamData(dataset['test'][:])
    data_predict = BeamData(dataset['train'][:])

    transformer.fit(data_train)
    preds = transformer.transform(data_predict)
    print(preds)
    print('done tfidf_transformer example')

